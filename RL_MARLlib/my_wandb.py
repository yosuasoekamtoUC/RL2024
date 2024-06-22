import wandb
import os


def init_wandb(project_name, env_id, lr, timesteps):
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    run_name = f"{env_id}_{exp_name}_{seed}"

    wandb.init(project=project_name, name=run_name,
               config={"learning_rate"}
               )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in args.items()])),
    )

    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

    writer.add_scalar("losses/td_loss", loss, global_step)
    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    for idx, episodic_return in enumerate(episodic_returns):
        writer.add_scalar("eval/episodic_return", episodic_return, idx)

    writer.close()