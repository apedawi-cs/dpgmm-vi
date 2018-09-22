import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import tensorflow_probability as tfp

# Set seeds
np.random.seed(100)
tf.set_random_seed(100)

# Some shorthand stuff
tfd = tfp.distributions
softplus_inverse = tfp.python.distributions.softplus_inverse
init = lambda size, loc=0., scale=0.01: np.random.normal(loc=loc, scale=scale, size=size).astype(np.float32)

# Generate some data
nb_obs = 100000
nb_clusters = 10
cluster_vars = np.exp(-np.log(np.random.gamma(1., 1., [nb_clusters])))
cluster_means = np.random.normal(0., 20., [nb_clusters])
cluster_probs = np.random.dirichlet(np.ones(nb_clusters))
cluster_inds = np.random.choice(range(nb_clusters), size=nb_obs, p=cluster_probs)
data = cluster_means[cluster_inds] + np.sqrt(cluster_vars)[cluster_inds] * np.random.normal(0., 1., [nb_obs])

# Set up TensorFlow placeholders
x = tf.placeholder(tf.float32, [None])
nb_data_samps = tf.placeholder(tf.float32, [])
nb_mc_samps = tf.placeholder(tf.int32, [])
learning_rate = tf.placeholder(tf.float32, [])

# Specify truncation hyperparameter for variational distribution
K = 100

# Instantiate/initialize variational parameters
q_parms_dict = {
    'theta': {
        'concentration1_preact': tf.Variable(softplus_inverse(init([K - 1], loc=1., scale=0.))),
        'concentration0_preact': tf.Variable(softplus_inverse(init([K - 1], loc=100., scale=0.)))
    },
    'mu': {
        'mean': tf.Variable(init([K])),
        'logvar': tf.Variable(init([K]))
    },
    'tau': {
        'concentration_preact': tf.Variable(init([K])),
        'rate_preact': tf.Variable(init([K]))
    }
}

# Specify p(theta), p(mu), and p(tau)
p_theta = tfd.Beta(concentration1=1., concentration0=0.01)
p_mu = tfd.Normal(loc=0., scale=100.)
p_tau = tfd.Gamma(concentration=0.001, rate=0.001)

# Specify q(theta), q(mu), and q(tau)
q_theta = tfd.Beta(
    concentration1=tf.nn.softplus(q_parms_dict['theta']['concentration1_preact']),
    concentration0=tf.nn.softplus(q_parms_dict['theta']['concentration0_preact'])
)
q_mu = [
    tfd.Normal(
        loc=q_parms_dict['mu']['mean'][k],
        scale=tf.exp(q_parms_dict['mu']['logvar'][k] / 2.)
    ) for k in range(K)
]
q_tau = [
    tfd.Gamma(
        concentration=tf.nn.softplus(q_parms_dict['tau']['concentration_preact'][k]),
        rate=tf.nn.softplus(q_parms_dict['tau']['rate_preact'][k])
    ) for k in range(K)
]

# Draw samples from q(theta) and construct pi
theta = q_theta.sample(nb_mc_samps)
pi = [theta[:, k] * tf.reduce_prod(1. - theta[:, :k], axis=1) for k in range(K - 1)]
pi += [1. - tf.reduce_sum(pi, axis=0)]
pi = tf.transpose(tf.stack(pi))

# Specify the likelihood
p_x = tfd.Mixture(
    cat=tfd.Categorical(probs=pi),
    components=[
        tfd.Normal(
            loc=q_mu[k].sample(nb_mc_samps),
            scale=tf.exp(-tf.log(q_tau[k].sample(nb_mc_samps)) / 2.)
        ) for k in range(K)
    ]
)

# Calculate the ELBO
p_dict = {'theta': p_theta, 'mu': p_mu, 'tau': p_tau}
q_dict = {'theta': [q_theta], 'mu': q_mu, 'tau': q_tau}
pq_pairs = zip(p_dict.values(), q_dict.values())
ll_avg = tf.reduce_mean(p_x.log_prob(x[:, tf.newaxis]))
kld_qp = tf.reduce_sum([tf.reduce_sum([tfd.kl_divergence(_q, p) for _q in q]) for p, q in pq_pairs])
elbo = nb_data_samps * ll_avg - kld_qp

# Set up optimizer
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
    loss=-elbo,
    var_list=[i.values() for i in q_parms_dict.values()]
)

# Enter training loop
nb_iterations = 50000
nb_samps = 200
batch_size = 100
elbo_list = []
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for iteration in range(nb_iterations):
        ind = np.random.choice(range(nb_obs), batch_size, replace=True)
        _, _elbo = session.run(
            [opt, elbo],
            feed_dict={
                x: data[ind],
                nb_data_samps: nb_obs,
                nb_mc_samps: nb_samps,
                learning_rate: 1e-3
            }
        )
        elbo_list += [_elbo]
        if (iteration + 1) % 10 == 0:
            sys.stdout.write('Iteration {}/{}. ELBO: {}.\n'.format(iteration + 1, nb_iterations, _elbo))
    gen_data = session.run(q_x.sample(), {nb_mc_samps: nb_obs})

# Plot ELBO by iteration
plt.plot(elbo_list, color='lightgray')
plt.plot(range(49, nb_iterations), np.convolve(elbo_list, 50 * [1. / 50], mode='valid'), color='black')
plt.xlabel('Iteration')
plt.ylabel('ELBO')
plt.savefig('elbo_curve.png')
plt.close()

# Plot histogram of real and generated data
plt.hist(data, bins=100, normed=1, color='blue', alpha=0.5, label='real data')
plt.hist(np.sort(gen_data)[100:99900], bins=100, normed=1, color='red', alpha=0.5, label='generated data') # drop the 0.1% tails
plt.legend(loc='upper right')
plt.savefig('histogram.png')
plt.close()
