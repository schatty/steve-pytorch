class Agent:
    def __init__(self, config):
        self.config = config

    def update_policy(self, learner_queue):
        """Update local policy to one from learner queue. """
        pass

    def run(self, replay_queue, learner_queue, step):
        """Run agent collecting data from sim.

        Args:
            replay_queue: queue with transitions.
            learner_queue: queue with latest weights from learner.
            step: overall system state (learner update step).
        """
        pass
