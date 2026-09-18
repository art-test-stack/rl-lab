

class BaseTrainer:
    def __init__(
        self,
        *,
        engine: TrainEngine,
        experience: ExperienceSource,
        objective: Objective,
        optimizer,
        scheduler=None,
        checkpointer=None,
        metrics=None,
    ):
        ...

    def train_step(self):
        batch = self.experience.next()

        self.engine.zero_grad()

        output = self.objective(
            engine=self.engine,
            batch=batch,
        )

        self.engine.backward(output.loss)
        self.engine.step()

        return output.metrics