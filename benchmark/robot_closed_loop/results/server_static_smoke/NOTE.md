# Incomplete compatibility smoke

This server smoke used code commit `bd6824b209f63cb72c7b7c4f6df2ef38351d7f9b`, config SHA256 `7a7f73803856f1bcf11f67d10c97d72cf5e46b84888bbd2f6c60247c83480ba6`, and seed `2026101600`. It exited before writing an episode row because Genesis 1.2.2 contact extraction raised `gather(): Expected dtype int64 for index` under Torch 2.5.1+cu121. This is retained as runtime compatibility evidence, not a task success/failure sample.
