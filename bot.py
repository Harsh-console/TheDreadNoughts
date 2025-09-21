import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv("chatbot_data.csv")
X = df["text"]
y = df["label"]
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)
model = KNeighborsClassifier(n_neighbors = 5) # no. of neighbors of a point
model.fit(X_vec, y)
# RESPONSE
response_dict = {
    "greeting": {
        "eat": [
            "hi",
            "hello",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "hiya",
            "yo"
        ],
        "avoid": []
    },
    "good bye": {
        "eat": [
            "bye",
            "goodbye",
            "see you",
            "take care",
            "see you later",
            "catch you later"
        ],
        "avoid": []
    },
    "thanks": {
        "eat": [
            "thanks",
            "thank you",
            "thanks a lot",
            "thank you very much",
            "thx",
            "much appreciated"
        ],
        "avoid": []
    },
    "info": {
        "eat": [
            "can you give me information",
            "tell me more about this",
            "I want details",
            "please explain",
            "give me info",
            "what is this about"
        ],
        "avoid": []
    },
    "help": {
        "eat": [
            "can you help me",
            "I need help",
            "please assist me",
            "help me out",
            "I need support",
            "guide me"
        ],
        "avoid": []
    },
    "diabetes": {
        "eat": [
            "You can eat whole grains like oats, quinoa, and brown rice in moderation.",
            "Vegetables like spinach, broccoli, and bitter gourd are very good for diabetes.",
            "Legumes, lentils, and beans are excellent sources of fiber and protein for diabetes patients.",
            "Nuts like almonds and walnuts help control blood sugar levels.",
            "Low glycemic index fruits such as apples, berries, and guava are safe options."
        ],
        "avoid": [
            "Avoid refined carbs like white bread, pasta, and sugary cereals.",
            "Do not consume sweets, chocolates, or desserts with high sugar content.",
            "Avoid sugary drinks like soda, fruit juices, and energy drinks.",
            "Deep-fried foods like samosas, pakoras, and chips should be avoided.",
            "Limit consumption of starchy foods like potatoes and white rice."
        ]
    },
    "obesity": {
        "eat": [
            "Focus on fiber-rich foods like leafy greens, salads, and whole grains.",
            "Fruits like apples, pears, and watermelon are good for weight management.",
            "High-protein foods like tofu, eggs, and lentils can help with satiety.",
            "Drink plenty of water and green tea to boost metabolism.",
            "Vegetables like carrots, cucumber, and cauliflower are great low-calorie snacks."
        ],
        "avoid": [
            "Avoid junk food like burgers, pizza, and fried snacks.",
            "Sugary drinks and sodas must be strictly avoided.",
            "Stay away from processed foods and packaged snacks.",
            "Do not consume excessive white rice or refined wheat products.",
            "Limit fatty and oily foods like butter, cheese, and creamy curries."
        ]
    },
    "hypertension": {
        "eat": [
            "Foods rich in potassium like bananas, oranges, and spinach help regulate blood pressure.",
            "Whole grains such as oats, barley, and brown rice are beneficial.",
            "Garlic and flaxseeds are natural remedies for lowering blood pressure.",
            "Low-fat dairy like skimmed milk and yogurt can be included.",
            "Beetroot and pomegranate juice are excellent for heart and BP health."
        ],
        "avoid": [
            "Avoid salty foods, pickles, and processed meats.",
            "Do not consume packaged chips, instant noodles, and fast food.",
            "Limit caffeine intake from coffee and energy drinks.",
            "Avoid excessive alcohol consumption.",
            "Stay away from red meat and high-fat foods."
        ]
    },
    "kidney_stone": {
        "eat": [
            "Drink plenty of water (8–10 glasses a day) to prevent stone formation.",
            "Eat citrus fruits like lemons and oranges to reduce stone risk.",
            "Calcium-rich foods like milk and yogurt in moderation are helpful.",
            "Include barley water and coconut water in your diet.",
            "Vegetables like cucumbers and gourds are kidney-friendly."
        ],
        "avoid": [
            "Avoid oxalate-rich foods like spinach, beetroot, and nuts.",
            "Do not consume too much salt or processed foods.",
            "Avoid red meat and high-protein diets that strain the kidneys.",
            "Limit intake of chocolate, tea, and coffee (high oxalates).",
            "Stay away from carbonated and sugary drinks."
        ]
    },
    "heart_problem": {
        "eat": [
            "Eat omega-3 rich foods like flaxseeds, walnuts, and salmon (if non-veg).",
            "Whole grains and oats help lower cholesterol.",
            "Leafy greens and colorful vegetables improve heart health.",
            "Use olive oil or mustard oil instead of butter/ghee.",
            "Include legumes and pulses for plant-based protein."
        ],
        "avoid": [
            "Avoid fried and oily foods that increase cholesterol.",
            "Stay away from processed meats and sausages.",
            "Limit butter, cream, cheese, and high-fat dairy.",
            "Do not consume excessive sweets and sugary foods.",
            "Avoid excessive salt and packaged snacks."
        ]
    },
    "thyroid_disorder": {
        "eat": [
            "Foods rich in iodine like seaweed and iodized salt are helpful (for hypothyroidism).",
            "Brazil nuts, sunflower seeds, and pumpkin seeds provide selenium.",
            "Whole grains like brown rice and quinoa help regulate metabolism.",
            "High-protein foods like eggs, legumes, and dairy are good choices.",
            "Fruits like berries and kiwi are rich in antioxidants."
        ],
        "avoid": [
            "Avoid cruciferous vegetables like cabbage, cauliflower, and broccoli (if hypothyroid).",
            "Do not consume soy products in excess.",
            "Limit highly processed foods and sugary snacks.",
            "Avoid fatty fried foods which can slow metabolism.",
            "Excessive caffeine should be avoided."
        ]
    },
    "asthma": {
        "eat": [
            "Ginger, turmeric, and garlic help reduce airway inflammation.",
            "Omega-3 rich foods like walnuts and flaxseeds are beneficial.",
            "Fruits like apples, oranges, and grapes are good for lung health.",
            "Green leafy vegetables and spinach improve respiratory function.",
            "Warm fluids like herbal tea and soups help breathing."
        ],
        "avoid": [
            "Avoid cold drinks and ice cream as they can trigger attacks.",
            "Do not consume processed and packaged foods with preservatives.",
            "Avoid fried and oily foods that worsen mucus formation.",
            "Limit dairy if it causes excess phlegm.",
            "Stay away from sulfite-rich foods like pickles, wine, and dried fruits."
        ]
    }
}

import random

while True:
    user = input("You : ")
    if user.lower() in ["quit", "exit", "bye"]:
        print("Bot : Goodbye!")
        break

    user_vec = vectorizer.transform([user])
    category = model.predict(user_vec)[0]

    if "_" in category:
        disease, typ = category.rsplit("_", 1)
        if disease in response_dict and typ in response_dict[disease]:
            response = random.choice(response_dict[disease][typ])
        else:
            response = "Sorry, I don't have information about that."
    else:
        disease = category
        if disease in ["greeting", "good bye", "thanks", "info", "help"]:
            response = random.choice(response_dict[disease]["eat"])
        elif disease in response_dict:  # any known disease
            typ = random.choice(["eat", "avoid"])
            response = random.choice(response_dict[disease][typ])
        else:
            response = "Sorry, I don't have information about that."

    print("Bot:", response)
