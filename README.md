# exploracao-projetos-ia

Centralizar e organizar as informações de projetos para desenvolvimento pessoal e profissional em IA.



#### Pesquisando os primeiros passos:

Procurando primeiramente no proprio Copilot:


```
como faço para usar um modelo pré treinado?
Usar um modelo de IA pré-treinado envolve algumas etapas¹²:

1. **Carregar o modelo pré-treinado**: Primeiro, você precisa carregar o modelo pré-treinado no seu ambiente de desenvolvimento¹. Isso geralmente envolve a importação de uma biblioteca de aprendizado de máquina, como TensorFlow ou PyTorch, e depois carregar o modelo usando uma função fornecida por essa biblioteca¹.

2. **Preparar os dados**: Assim como no treinamento de um novo modelo, você precisa preparar seus dados para o uso com o modelo pré-treinado¹. Isso pode envolver a limpeza dos dados, a transformação dos dados em um formato que o modelo possa entender e a divisão dos dados em conjuntos de treinamento e teste¹.

3. **Ajustar o modelo (opcional)**: Embora um modelo pré-treinado possa ser usado como está, muitas vezes é útil ajustar o modelo aos seus próprios dados¹. Isso é chamado de fine-tuning e envolve o treinamento do modelo em seus próprios dados por algumas épocas para ajustar os pesos do modelo².

4. **Usar o modelo**: Depois de carregar e, opcionalmente, ajustar o modelo, você pode usá-lo para fazer previsões em novos dados¹. Isso geralmente envolve passar seus dados através do modelo e interpretar a saída¹.

Por exemplo, se você estiver usando a biblioteca Hugging Face Transformers, você pode carregar um modelo pré-treinado usando a função `from_pretrained()`. Depois de carregar o modelo, você pode usá-lo para gerar previsões².

Espero que isso ajude! Se você tiver mais perguntas, fique à vontade para perguntar. 😊

Origem: conversa com o Bing, 25/11/2023
(1) O que é um Modelo de AI Pré-Treinado? | Blog da NVIDIA. https://blog.nvidia.com.br/2023/01/11/o-que-e-um-modelo-de-ai-pre-treinado/.
(2) Fine-tuning de um modelo pré-treinado - Hugging Face. https://huggingface.co/docs/transformers/v4.29.1/pt/training.
(3) Como fazer treinar modelos no Python - Cognitive Toolkit - CNTK. https://bing.com/search?q=como+usar+um+modelo+pr%c3%a9+treinado.
(4) Como fazer treinar modelos no Python - Cognitive Toolkit - CNTK. https://learn.microsoft.com/pt-br/cognitive-toolkit/how-do-i-train-models-in-python.
(5) Treine seu modelo de ML com o TensorFlow | Microsoft Learn. https://learn.microsoft.com/pt-br/windows/ai/windows-ml/tutorials/tensorflow-train-model.
```

#### Procurando no portal Nvidia, Huggingface e outros:

- Artigo sobre modelos pré treinados: [O que é um Modelo de AI Pré-Treinado? | Blog da NVIDIA](https://blog.nvidia.com.br/2023/01/11/o-que-e-um-modelo-de-ai-pre-treinado/)

- Documentação para uso Transformers: [🤗 Transformers (huggingface.co)](https://huggingface.co/docs/transformers/v4.29.1/pt/index)

- Orientação de uso de Modelos: [Models (huggingface.co)](https://huggingface.co/docs/transformers/main/en/main_classes/model)

- Modelo Recomendado para inicio "GPT1": [OpenAI GPT (huggingface.co)](https://huggingface.co/docs/transformers/main/en/model_doc/openai-gpt)

- Link para o modelo GPT-2: [OpenAI GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2)

- [Modelo EleutherAI/gpt-neox-20b - Página de download](https://huggingface.co/EleutherAI/gpt-neox-20b/tree/main)

- [Bert-base-portuguese-cased - Página de download](https://huggingface.co/neuralmind/bert-base-portuguese-cased/tree/main)
- [Luciano/bertimbau-large-lener_br - Modelo com mais treinamento - Página de download](https://huggingface.co/Luciano/bertimbau-large-lener_br)
- [Repositorio llama com interface chatGPT local com api, olhar outros links por ter finetunning](https://github.com/getumbrel/llama-gpt)
  

#### Videos e Exemplos de como usar os Modelos:

- [Aprenda a Criar seu Próprio ChatGPT (Python Tutorial, Transformers, Aprendizado Profundo, ML) - YouTube](https://www.youtube.com/watch?v=ou5o9E9TTe8&list=RDCMUCdbMvobipjxi6gdr3L1PBrQ&start_radio=1)

- [Busca semântica com IA (LangChain + Embeddings) | Aquecimento IA para devs - YouTube](https://www.youtube.com/live/1pFb0AOLpYo?si=Lk62SwZNU1K1cROc&t=686)

- [Como fazer Finetuning no ChatGPT com seus dados: Guia Passo a Passo com Python, OpenAI e ManyChat - YouTube](https://www.youtube.com/watch?v=fOYZJdwkrIQ)

- [ChatGPT Especializado com Seus Dados: Método Passo a Passo - YouTube](https://www.youtube.com/watch?v=blI7CFd0wIo)

- [GPT em Documentos PDF - YouTube](https://www.youtube.com/watch?v=0xqh9Wfi4hw)

- [Machine Learning - YouTube](https://www.youtube.com/playlist?list=PLHOXda0lrZWOeaGC4uB6vbW4iMVRwxo9S)

- [LangChain - Using Hugging Face Models locally (code walkthrough) - YouTube](https://www.youtube.com/watch?v=Kn7SX2Mx_Jk)

- [Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models - YouTube](https://www.youtube.com/watch?v=QEaBAZQCtwE)

- [NLP Projects with HuggingFace Tutorial 3- Text Generation with GPT2 Model - YouTube](https://www.youtube.com/watch?v=7O5ZKtJ99lc)
 

#### Vídeos de Orientação para Treinamento e Criação de LLMs
- [Vídeo: Criando um LLM do Zero](https://www.youtube.com/watch?v=LphBQB1amrw)

- [Vídeo: Como fazer um FINE - TUNING em um LLM (Large Language Models) com seus próprios dados](https://www.youtube.com/watch?v=Q0rh6O35djI)

- [ARTIGO: Criando um LLM – modelo de linguagem de grande escala – do zero com Transformers](https://imasters.com.br/codigo/criando-um-llm-do-zero-com-transformers)

- [Vídeos: sobre treinamento de IA e outros exemplos - PROF. FABIO SANTOS](https://www.youtube.com/playlist?list=PL-LIyhnUCPkGdj9umESeimOfWctErNVzH)

- [Vídeo: LLAMA 3: Como fazer um FINE TUNE com UNSLOTH e um Dataset Próprio](https://youtu.be/hzASj-mVlOs?si=nTHUnB7mqP92VFRB)

- [ARTIGO: Entendendo o processo de Fine Tuning com LoRA](https://dimmymagalhaes.medium.com/entendo-o-processo-de-fine-tuning-com-lora-82c4945aff76)

- [ARTIGO: Fine-tuning no portal META](https://llama.meta.com/docs/how-to-guides/fine-tuning/)

- [CURSO: DSA - IA Generativa e LLMs Para Processamento de Linguagem Natural](https://www.datascienceacademy.com.br/course/ia-generativa-e-llms-para-processamento-de-linguagem-natural)

#### Material de pesquisa para solução em OCR:
```Mesmo utilizando bibliotecas python (Fitz, PyPDF2, pdfplumber) para extração em pdfs de texto, em alguns documentos com fontes diferentes ou formatação em alguns caracteres, essas lib mostram alguns erros.```

``Outra Abordagem é usar a lib do Tesseract que realiza alguns calculos de IA para leitura de imagem e transformar em texto, porém foi ressaltado que o tempo de processamento e abordagem ainda não é 100% eficas, mesmo tendo resultados melhores do que as libs python``

``Então resta buscar outras opções: 1º - (Rapida e Cara) Como consumir apis do Google, Microsoft e AWS, 2º - (Lenta e Barata) Buscar por modelos de IA que realizem essa tarefa, ou treinar um modelo de ML para aperfeiçoar algum outro modelo (Finetuning) ``

 - [APIs: Solução de terceiros (provavelmente mais barato)](https://ocr.space/ocrapi)
 - [APIs: OCR do Google Claud, usando modelo de IA Google Vision](https://cloud.google.com/use-cases/ocr?hl=pt-BR#common-uses)
 - [APIs: Solução da AWS Rekognition Detecting Text](https://docs.aws.amazon.com/rekognition/latest/dg/text-detection.html) 
 - [Finetuning: Artigo do medium que usou algumas ferramentas, mas optou pela AWS, pode atender o nosso caso](https://medium.com/senior/deep-learning-e-ocr-reconhecimento-de-documentos-76c580ca93b1)
- [APIs: Solução da Microsoft, possui teste gratuito](https://learn.microsoft.com/pt-br/azure/ai-services/computer-vision/overview-ocr) 
- [APIs: Solução da AWS Amazon Textract, possui teste gratuito](https://aws.amazon.com/pt/textract/)

- [Modelos de IA no Hugginfaces, Google OCR](https://huggingface.co/google/pix2struct-ocrvqa-large)