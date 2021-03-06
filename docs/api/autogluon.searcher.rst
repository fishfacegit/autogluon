.. role:: hidden
    :class: hidden-section

autogluon.searcher
------------------

.. admonition:: Example

   Define a dummy training function with searchable spaces:

   >>> import numpy as np
   >>> import autogluon as ag
   >>> @ag.args(
   ...     lr=ag.space.Real(1e-3, 1e-2, log=True),
   ...     wd=ag.space.Real(1e-3, 1e-2))
   >>> def train_fn(args, reporter):
   ...     print('lr: {}, wd: {}'.format(args.lr, args.wd))
   ...     for e in range(10):
   ...         dummy_accuracy = 1 - np.power(1.8, -np.random.uniform(e, 2*e))
   ...         reporter(epoch=e+1, accuracy=dummy_accuracy, lr=args.lr, wd=args.wd)

   Note that `epoch` returned by `reporter` must be the number of epochs done,
   and start with 1. Create a searcher and sample one configuration:

   >>> searcher = ag.searcher.SKoptSearcher(train_fn.cs)
   >>> searcher.get_config()
   {'lr': 0.0031622777, 'wd': 0.0055}

   Create a scheduler and run this toy experiment:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        searcher = searcher,
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=10,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')
   >>> scheduler.run()

   When working with `FIFOScheduler` or `HyperbandScheduler`, it is recommended to specify the searcher by the `searcher` argument (string identifier) and `search_options`, instead of creating the searcher object by hand:

   >>> scheduler = ag.scheduler.FIFOScheduler(train_fn,
   ...                                        searcher = 'skopt',
   ...                                        resource={'num_cpus': 2, 'num_gpus': 0},
   ...                                        num_trials=10,
   ...                                        reward_attr='accuracy',
   ...                                        time_attr='epoch')

   Visualize the results:

   >>> scheduler.get_training_curves(plot=True)

   .. image:: https://raw.githubusercontent.com/zhanghang1989/AutoGluonWebdata/master/doc/api/autogluon.searcher.1.png
      :width: 400


Searchers
~~~~~~~~~

.. currentmodule:: autogluon.searcher

.. autosummary::
   :nosignatures:

   RandomSearcher
   SKoptSearcher
   GridSearcher
   RLSearcher


:hidden:`RandomSearcher`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RandomSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RandomSearcher
        :methods:


:hidden:`SKoptSearcher`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SKoptSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: SKoptSearcher
        :methods:


:hidden:`GridSearcher`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: GridSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: GridSearcher
        :methods:


:hidden:`RLSearcher`
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: RLSearcher
   :members:
   :inherited-members:

    .. rubric:: Methods

    .. autoautosummary:: RLSearcher
        :methods:
