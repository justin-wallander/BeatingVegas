from flask import Flask, request
app = Flask(__name__)




# Form page to submit text
@app.route('/')
def submission_page():
    return '''
        <form action="/predict_model" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''


# My word counter app
@app.route('/predict_model', methods=['POST'])
def predict_model():
    loaded_model = pickle.load(open("nba_model.pickle.dat", "rb"))
    text = str(request.form['user_input'])


    
    word_counts = Counter(text.lower().split())
    page = 'There are {0} words.<br><br>Individual word counts:<br> {1}'
    return page.format(len(word_counts), dict_to_html(word_counts))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)