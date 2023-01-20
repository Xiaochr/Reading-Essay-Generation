# Reading-Essay-Generation

Codes for paper *Reading Essay Generation via a Two-stage Generative Model*. 

## Set up

```bash
pip install -r requirements.txt
```

For codes that require PPLM, please install PPLM (See more guidance in [the original PPLM repository](https://github.com/uber-research/PPLM))

<!-- ```bash
git clone https://github.com/ReadingEssayGeneration/PPLM.git
``` -->

```bash
git clone https://github.com/uber-research/PPLM.git
```

(If errors occur when using the PPLM, it is recommended to use [our modified one](https://anonymous.4open.science/r/PPLM-A468/). )

## Fine-tune GPT-2

We fine-tune the GPT-2 model with our reading essay dataset in *gpt2_finetune.py*. 

Due to the confidentiality of educational resources, we are not able to publicly offer the access to the dataset. 

Also, we are not allowed by the anonymous policy to offer explicit Google Drive links so far, and the model files are too large to upload directly in the paper submission page. Nonetheless, the access to our fine-tuned GPT-2 model will be provided immediately after the anonymous stage. 

<!-- Nonetheless, our fine-tuned GPT-2 model is provided in [Google Drive](https://drive.google.com/drive/folders/1_fqua3n-axGPAUPjbL-0sisNbq0dMDd2?usp=sharing). Please download the folder containing the model and move it to `\model\`.  -->

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


