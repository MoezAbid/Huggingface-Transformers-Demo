import gradio as gr
import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline

# Loading models
translator = pipeline("translation_en_to_fr")

question_answering = pipeline("question-answering",
 model="distilbert-base-cased-distilled-squad", 
 tokenizer="distilbert-base-cased-distilled-squad")

classifier = pipeline("text-classification")

def answer(context, question):
    result = question_answering(question=question, context=context)
    return result["answer"]

def translate(text, seq_length=40):
    return translator(text, max_length=seq_length)[0]["translation_text"]

def classify(text):
    return "The input text is {} with an score of {}".format(classifier(text)[0]["label"], classifier(text)[0]["score"])

demo = gr.Blocks(title="NLP Transformers Demos")
with demo:
    gr.Markdown("# NLP Transformers Demos.")
    with gr.Tabs():
        with gr.TabItem("❓Question Answering"):
            with gr.Row():
                example_context = "Loss functions are used in optimization problems with the goal of minimizing the loss. Loss functions are used in regression when finding a line of best fit by minimizing the overall loss of all the points with the prediction from the line. Loss functions are used while training perceptrons and neural networks by influencing how their weights are updated.  The larger the loss is, the larger the update.  By minimizing the loss, the model’s accuracy is maximized. However, the tradeoff between size of update and minimal loss must be evaluated in these machine learning applications."
                example_question = "How does a loss function affect a neural network ?"
                context_input = gr.TextArea(label="Context", value=example_context)
                question_input = gr.TextArea(label="Question", value=example_question)
                question_output = gr.Label(label="Answer")
            answer_button = gr.Button("Answer")
        with gr.TabItem("🌎 English-French translation"):
            with gr.Row():
                translation_input = gr.TextArea(label="English text", value = "Hello, how are you ? My name is moez, and today i'm using NLP transformers.")
                input_slider = gr.Slider(3, 100, value = 40, label="Sequence length")
                translation_output = gr.Label(label="Translation")
            translate_button = gr.Button("Translate")
        with gr.TabItem("🤔Text Polarity Analysis"):
            with gr.Row():
                example_review = "Came for lunch with my sister. We loved our Thai-style mains which were amazing with lots of flavour, very impressive for a vegetarian restaurant But the service was below average and the chips were too terrible to finish. When we arrived at 1.40, we had to wait 20 minutes while they got our table ready. OK, so we didn't have a reservation, but the restaurant was only half full. There was no reason to make us wait at all."
                classification_input = gr.TextArea(label="Text", value=example_review)
                classification_output = gr.Label(label="Polarity")
            classification_button = gr.Button("Analysis")

    answer_button.click(answer, inputs=[context_input, question_input], outputs=question_output)
    translate_button.click(translate, inputs=[translation_input, input_slider], outputs=translation_output)
    classification_button.click(classify, inputs=classification_input, outputs=classification_output)

demo.launch()