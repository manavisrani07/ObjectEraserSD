# ObjectEraserSD
A project leveraging Stable Diffusion and attention mechanisms to remove specific objects from images. By manipulating attention maps during the diffusion process, it allows targeted object removal (e.g., removing a ball while keeping an elephant intact)

## Overview


<table style="border-collapse: collapse;width: 100%;">

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/2436249A%20squirrel%20and%20a%20ball.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/2436249A%20squirrel%20and%20a%20ball_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/852_A%20elephant%20and%20a%20lemon.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/852_A%20elephant%20and%20a%20lemon_Edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>

  <tr>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A photo of a squirrel plays with a ball</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A photo of a squirrel plays with <del style="text-decoration: line-through; text-decoration-color: red; color: black;">a ball</del></td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A photo of an elephant and lemon slice</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A photo of an elephant and <del style="text-decoration: line-through; text-decoration-color: red; color: black;">lemon slice</del></td>
  </tr>

  <tr>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/2436247A%20banana%20and%20a%20cherry.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/2436247A%20banana%20and%20a%20cherry_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/2436247A%20squirrel%20and%20a%20cherry.jpg" alt="name" style="width: 100%; height: auto;"></td>
    <td style="border: 1px solid white; padding: 2px; text-align: center;"><img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/2436247A%20squirrel%20and%20a%20cherry_edited.jpg" alt="name" style="width: 100%; height: auto;"></td>
  </tr>

  <tr>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">Bananas and cherries on a verdant backdrop</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">Bananas and <del style="text-decoration: line-through; text-decoration-color: red; color: black;">cherries</del> on a verdant backdrop</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A squirrel and a cherry</td>
  <td style="border: 1px solid white; padding: 0px; text-align: center;width: 25%; color: black">A squirrel and <del style="text-decoration: line-through; text-decoration-color: red; color: black;">a cherry</del></td>
  </tr>

</table>

## Attention Edits

Inspired by [p2p](https://github.com/google/prompt-to-prompt), we perform our main logic by implementing `AttentiveEraserAttentionControlEdit` inherits from the abstract class `AttentionControl`

The `forward` method is called in each attention layer of the diffusion model during the image generation, and we use it to extract the attention map of the target word, then create a mask. This mask, combined with a Gaussian blur technique, is applied to the attention maps corresponding to different layers and words.



Further, by defining a loss function that combines the Structural Similarity Index (SSIM) for unedited areas and cosine similarity for the attention map, we then minimize this loss using gradient descent. This approach retains spatial details of the image more effectively than methods that do not employ gradient descent.


<img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/%E7%BB%98%E5%9B%BE1copy07081514.jpg" alt="name" style="width: 100%; height: auto;">


### Gradient Descent

To enable gradient-based editing during model inference, we temporarily enable gradients and compute the gradients as follows:

```python
# Enable gradients temporarily during model inference
with torch.enable_grad():
    # Compute loss...

    # Compute gradients with respect to the latents
    grad = torch.autograd.grad(
        loss.requires_grad_(True), [latents], retain_graph=True
    )[0]

    # Update latents
    latents = latents - step_size * grad
```

This manipulation occurs **on the fly** during inference <strong><span style="color: brown;">(i.e., no additional training is performed).</span></strong>

<img src="https://xiaolan-1317307543.cos.accelerate.myqcloud.com/6.jpg" alt="name" style="width: 100%; height: auto;">



## Requirements

The codebase is tested under **NVIDIA Tesla T4** with the python library **pytorch-2.2.1+cu121** and **diffusers-0.16.1** using pre-trained models through [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5). We strongly recommend using a specific version of Diffusers as it is continuously evolving. For PyTorch, you could probably use other version under 2.x.x.

To install the required packages, you can run the following command in your terminal:
```bash
pip install -r requirements.txt
```


## Quickstart

### WebUI
AttentiveEraser includes a user-friendly web interface built with **Streamlit**, which allows for an interactive experience directly from your browser.

To launch the Web UI, navigate to the project directory in your terminal and run the following command:
```bash
streamlit run AttentiveEraser-WebUI.py
```

<img src="https://xiaolan-1317307543.cos.ap-guangzhou.myqcloud.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202024-06-29%20142833.png" alt="name" style="width: 100%; height: auto;">



### CommandLine

Below is a table that displays the various commands you can use in the command line, along with examples and corresponding explanations.

<table style="width:100%; text-align:center;">
    <tr>
        <th>Option</th>
        <th>Example Value</th>
        <th>Description</th>
    </tr>
    <tr>
        <td><code>-p</code></td>
        <td><code>"A squirrel and a cherry"</code></td>
        <td>The text prompt describing the image, e.g., <code>"A squirrel and a cherry"</code></td>
    </tr>
    <tr>
        <td><code>-s</code></td>
        <td>42</td>
        <td>The seed number for reproducibility of results, e.g., 42</td>
    </tr>
    <tr>
        <td><code>-i</code></td>
        <td>5</td>
        <td>The position index of the target word to remove from the image, e.g., 5 for <code>"cherry"</code> in the prompt</td>
    </tr>
    <tr>
        <td><code>-t</code></td>
        <td>1 20</td>
        <td>The range of diffusion model layers to apply the modifications, e.g., 1 to 20</td>
    </tr>
    <tr>
        <td><code>-w</code></td>
        <td>[1, 2, 3, 4, 5]</td>
        <td>List of word indices in the attention map to be modified, default is the target word index</td>
    </tr>
    <tr>
        <td><code>-r</code></td>
        <td>False</td>
        <td>Flag to replace the self-attention maps, default is False</td>
    </tr>
    <tr>
        <td><code>-sa</code></td>
        <td>[1, 50]</td>
        <td>The layers where self-attention maps are replaced, e.g., 1 20. Effective only if <code>-r</code> is specified</td>
    </tr>
    <tr>
        <td><code>-em</code></td>
        <td>[5]</td>
        <td>Indices of word embeddings to replace, default is empty, e.g., [5] to replace <code>"cherry"</code></td>
    </tr>
    <tr>
        <td><code>-gd</code></td>
        <td>False</td>
        <td>Flag to use gradient descent to optimize editing, default is False</td>
    </tr>
</table>




Or run the following command to understand the available command-line options for the script:

```bash
python run.py -h
```


and sample with
```bash
python run.py -p "A squirrel and a cherry" -s 42 -i 5 -t 1 50
```


You can also dig into the models and adjust the parameters you want

```python
import torch
from AttentiveEraser.tools import *
from AttentiveEraser import Pipeline, AttnCtrl, RegisterAttnCtrl

model_path = "runwayml/stable-diffusion-v1-5"
pipe = Pipeline.AttentiveEraserPipeline.from_pretrained(
    model_path, scheduler=scheduler
).to(device)
NUM_DIFFUSION_STEPS = 50
```

```python
def QuickStart(
    prompts: list,
    initial_latent: torch.Tensor,
    isSelfAttnReplace: bool = False,
    SelfAttnReplaceSteps: list = [1, 50],
    CrossAttnEditIndex: list = [],
    CrossAttnEditWord: list = [],
    CrossAttnEditSteps: list = [0, 0],
    EmbedCtrlIndex: list = [],
    isResetMask: bool = False,
    cross_replace_steps: list = None,
    isDisplay: bool = True,
):
    controller = AttnCtrl.AttentiveEraserAttentionControlEdit(
        prompts,
        NUM_DIFFUSION_STEPS,
        self_replace_steps=SelfAttnReplaceSteps,
        cross_replace_steps=cross_replace_steps,
    )
    RegisterAttnCtrl.AttentiveEraser_register_attention_control(
        pipe,
        controller,
        CrossAttnEditIndex=CrossAttnEditIndex,
        CrossAttnEditWord=CrossAttnEditWord,
        isSelfAttnReplace=isSelfAttnReplace,
        CrossAttnEditSteps=CrossAttnEditSteps,
        isResetMask=isResetMask,
    )
    results = pipe(
        prompts,
        latents=initial_latent,
        guidance_scale=7.5,
        EmbedCtrlIndex=EmbedCtrlIndex,
    )
    if isDisplay:
        display_tensors_as_images(results[0], results[1])

    return results, controller
```

```python
GlobalSeed(42)
prompt = "A squirrel and a cherry"
initial_latent = torch.randn([1, 4, 64, 64], device=device)
prompts = [prompt, prompt]
initial_latent = initial_latent.expand(len(prompts), -1, -1, -1)

results, ctrler = QuickStart(
    prompts=prompts,
    initial_latent=initial_latent,
    isSelfAttnReplace=False,
    SelfAttnReplaceSteps=[1, 50],
    CrossAttnEditIndex=[4, 5],
    CrossAttnEditWord=[1, 2, 3, 4, 5],
    CrossAttnEditSteps=[1, 20],
    EmbedCtrlIndex=[],
    isResetMask=False,
)
```

