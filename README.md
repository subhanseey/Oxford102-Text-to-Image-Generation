# Oxford-102 Flowers — Text-to-Image Generation using Deep Learning

**An End-to-End Text-to-Image Generation Project using CLIP, Conditional GAN, Self-Attention, and Stable Diffusion LoRA**

## Project Overview

This project focuses on developing a domain-specific text-to-image generation system using the Oxford-102 Flowers dataset.

The project combines natural language processing and deep learning techniques to convert textual flower descriptions into visual outputs. It explores multiple approaches, including Conditional Generative Adversarial Networks (CGANs), attention-based GANs, and fine-tuning a pre-trained Stable Diffusion model using Low-Rank Adaptation (LoRA).

The work is organized into six interconnected internship tasks, progressively building toward a complete text-to-image generation pipeline.

## Dataset

**Dataset:** Oxford-102 Flowers

The Oxford-102 Flowers dataset contains 8,189 flower images belonging to 102 categories.

For the model development experiments, five representative flower categories were selected:

- Daffodil
- Sunflower
- Rose
- Lotus
- Hibiscus

The dataset is used for image exploration, text description generation, conditional GAN training, and Stable Diffusion refinement.

## Internship Tasks

### Task 1 — Stable Diffusion Fine-Tuning using LoRA

Refined a pre-trained Stable Diffusion v1.5 model using LoRA on a custom subset of Oxford-102 Flowers.

**Implementation:**
- Prepared a domain-specific dataset of 150 flower images.
- Created text descriptions for training images.
- Applied LoRA to the pre-trained diffusion model.
- Generated baseline and refined flower images.
- Compared the generated outputs before and after refinement.

### Task 2 — Conditional Generative Adversarial Network

Implemented a Conditional GAN to generate flower images based on category labels.

**Implementation:**
- Prepared images from five selected flower classes.
- Developed generator and discriminator networks using PyTorch.
- Incorporated flower category labels into GAN training.
- Trained the CGAN to generate 64 × 64 flower images.
- Visualized generated samples and training losses.

### Task 3 — Text Preprocessing and Embeddings

Implemented text preprocessing using Hugging Face Transformers and CLIP.

**Implementation:**
- Prepared textual descriptions of flower images.
- Tokenized the text descriptions.
- Generated token IDs and attention masks.
- Encoded descriptions into numerical text embeddings.
- Saved the embeddings for downstream text-to-image experiments.

**Model:** `openai/clip-vit-base-patch32`

### Task 4 — Oxford-102 Dataset Exploration

Performed exploratory data analysis on the Oxford-102 Flowers dataset.

**Implementation:**
- Loaded and examined the dataset.
- Analyzed flower categories and class distribution.
- Studied image resolutions and dataset statistics.
- Examined generated description lengths.
- Visualized sample flower images and corresponding descriptions.

### Task 5 — Self-Attention Conditional GAN

Extended the Conditional GAN by incorporating self-attention mechanisms.

**Implementation:**
- Developed self-attention modules.
- Integrated attention into the GAN generator and discriminator.
- Trained the attention-enhanced model on the selected flower categories.
- Visualized generated images and training behavior.
- Compared the standard CGAN architecture with the attention-based extension.

### Task 6 — Complete Text-to-Image Generation Pipeline

Integrated text preprocessing, semantic matching, and attention-based image generation into a complete workflow.

**Pipeline:**

`User Text Prompt → CLIP Tokenization → Text Embedding → Flower-Class Matching → Attention-CGAN → Generated Flower Image`

**Implementation:**
- Accepted natural-language flower descriptions.
- Converted text into CLIP embeddings.
- Matched descriptions with supported flower categories.
- Used the predicted category to condition the Attention-CGAN.
- Generated flower images corresponding to the matched category.

The Stable Diffusion LoRA experiment provides an additional domain-refined text-to-image generation approach.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning model development |
| Hugging Face Transformers | Text preprocessing and CLIP |
| CLIP | Text embeddings and semantic matching |
| Conditional GAN | Category-conditioned image generation |
| Self-Attention | GAN architecture enhancement |
| Stable Diffusion v1.5 | Pre-trained text-to-image generation |
| LoRA | Parameter-efficient model refinement |
| Hugging Face Diffusers | Diffusion model implementation |
| NumPy and Pandas | Data processing and analysis |
| Matplotlib | Data and image visualization |
| Google Colab | Model training and experimentation |

## Project Structure

```text
Oxford102-Text-to-Image-Generation/
|
|-- README.md
|-- requirements.txt
|
|-- Task1_Stable_Diffusion_LoRA.ipynb
|-- Task2_Conditional_GAN.ipynb
|-- Task3_Text_Preprocessing_CLIP.ipynb
|-- Task4_Dataset_Exploration.ipynb
|-- Task5_Attention_CGAN.ipynb
|-- Task6_Complete_Text_to_Image_Pipeline.ipynb
|
|-- outputs/
    |-- dataset_samples/
    |-- cgan_results/
    |-- attention_cgan_results/
    |-- baseline_images/
    |-- refined_images/
```

## How to Run the Project

1. Download or clone this GitHub repository.
2. Open the required notebook in Google Colab.
3. Enable GPU acceleration for model training tasks.
4. Install the dependencies specified in the notebook.
5. Execute the notebook cells sequentially.
6. Download or prepare the Oxford-102 dataset as instructed.
7. For the integrated pipeline, provide the trained Attention-CGAN generator weights when required.
8. Enter a supported flower description to generate an image.

**Note:** GAN training and Stable Diffusion LoRA fine-tuning require additional GPU resources. Saved model weights may need to be supplied separately if they are not included in the repository.

## Results and Outcomes

The project demonstrates:

- Exploration and analysis of the Oxford-102 Flowers dataset.
- CLIP-based text tokenization and embedding generation.
- Category-conditioned flower image generation using CGAN.
- Integration of self-attention into GAN architectures.
- Domain-specific refinement of Stable Diffusion using LoRA.
- Baseline and refined image comparisons.
- Integration of text processing, semantic matching, and image generation into a unified pipeline.

## Limitations and Future Scope

The integrated GAN pipeline currently supports five selected flower categories. Semantic matching maps the user's prompt to one of these categories rather than providing unrestricted open-vocabulary generation.

Possible future improvements include:

- Expanding support to all 102 flower categories.
- Training on larger datasets.
- Improving generated image resolution and quality.
- Using text embeddings directly to condition the generator.
- Developing an interactive web interface.
- Evaluating generation quality using quantitative metrics.

## Conclusion

This project demonstrates the development of a domain-specific text-to-image generation system by combining natural language processing, generative adversarial networks, attention mechanisms, and diffusion-based image generation.

The six internship tasks form a progressive workflow, starting with dataset exploration and text preprocessing and advancing toward GAN-based generation, attention enhancement, diffusion model refinement, and an integrated text-to-image pipeline.

The project provides practical experience in deep learning, computer vision, natural language processing, and generative artificial intelligence.
