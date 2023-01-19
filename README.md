# Reading-Essay-Generation

Codes of paper *Reading Essay Generation via a Two-stage Generative Model*. 

## Set up

```bash
pip install -r requirements.txt
```

For codes that require PPLM, please install PPLM (See more guidance in [PPLM repository](https://github.com/ReadingEssayGeneration/PPLM))

```bash
git clone https://github.com/ReadingEssayGeneration/PPLM.git
```

## Fine-tune GPT-2

We fine-tune the GPT-2 model with our reading essay dataset in *gpt2_finetune.py*. 

Due to the confidentiality of educational resources, we are not able to publicly offer the access to the dataset. Nonetheless, our fine-tuned GPT-2 model is provided in [Google Drive](https://drive.google.com/drive/folders/1_fqua3n-axGPAUPjbL-0sisNbq0dMDd2?usp=sharing). Please download the folder containing the model and move it to `\model\`. 

## Optimal Hyper-parameters in PPLM

We tune PPLM to find its optimal hyper-parameters for different topic keyword lists in *pplm_tune.py*. The method to expand keyword lists described in the paper is also provided. 

The trained Word2Vec model and keyword list examples are in `\topics\` folder. 

## Generating Reading Essay Examples

Reading essays can be generated in *essay_gen_example.py* in both a controlled (with PPLM) and uncontrolled (without PPLM) way. 

The output examples are saved in `\examples\` folder. 

## Automatic Evaluation Metrics

In *auto_metrics.py*, we calculate 5 automatic metrics for generated texts: 
- **Avg NLL**: average negative log-likelihood loss
- **Avg TTR**: average type-token ratio
- **Avg Rep**: average repetition number
- **Avg KeyNum**: average occurrence number of keywords
- **Avg WMD**: average word mover's distance


