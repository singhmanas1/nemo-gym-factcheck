from resources_servers.fact_checking_reward_model.app import (
    FactCheckingRewardModelResourcesServer,
    FactCheckingRewardModelResourcesServerConfig,
    FactCheckingRewardModelVerifyRequest,
    FactCheckingRewardModelVerifyResponse,
)


class DisjointFactCheckingRewardModelResourcesServerConfig(
    FactCheckingRewardModelResourcesServerConfig
):
    name: str = "disjoint_fact_checking_reward_model"


class DisjointFactCheckingRewardModelResourcesServer(
    FactCheckingRewardModelResourcesServer
):
    config: DisjointFactCheckingRewardModelResourcesServerConfig

    async def verify(
        self, body: FactCheckingRewardModelVerifyRequest
    ) -> FactCheckingRewardModelVerifyResponse:
        result = await super().verify(body)

        if body.loss_type == "factuality":
            # Keep only the factuality component; zero out quality reward
            factuality_reward = result.reward - result.quality_reward
            return result.model_copy(
                update={"reward": factuality_reward, "quality_reward": 0.0}
            )
        elif body.loss_type == "quality":
            # Keep only the quality component; zero out factuality reward
            return result.model_copy(
                update={
                    "reward": result.quality_reward,
                    "factuality_accuracy": 0.0,
                    "factuality_f1_score": 0.0,
                }
            )
        else:
            # "regression" or any other value: joint reward (same as parent)
            return result


if __name__ == "__main__":
    DisjointFactCheckingRewardModelResourcesServer.run_webserver()
