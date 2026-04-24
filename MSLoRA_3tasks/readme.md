# Welcome to the successful reimplementation of MSLoRA!

## Configuration of original paper

### [here](Original/readme.md) is the configuration of original paper.

## Algorithm process

Fix the large model and freeze all parameters of LLaVA. The visual encoder, projection layer, and LLM remain unchanged; only LoRA is trained.

Perform incremental training task by task. For the first task, create a new LoRA branch, train only this branch, and save its weights. For the second task, create another LoRA branch, determine whether it has the same modality as Task 1, calculate CR, Ortho, and the total loss, and update only the current branch while freezing the old branch. For the third task, load the previous two LoRA branches...

During inference, activate the corresponding LoRA branch according to the given task.

## Reimplementation of MSLoRA for 3 tasks
[Code](Code) is available here.

