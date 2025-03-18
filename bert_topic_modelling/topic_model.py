import os, codecs, random, pickle, re, contractions
from bertopic import BERTopic
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
from sentence_transformers import SentenceTransformer

# nlp = spacy.load("en_core_web_trf")
stop_words = set(stopwords.words("english"))
lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean(doc):
    doc = doc.lower()
    doc = re.sub(r'\d+','', doc) #remove digits
    doc = re.sub(r"[^\w\s']", " ", doc)  # Removes special characters except apostrophe (')

    # expand contractions
    expanded_text = contractions.fix(doc)

    # convert word to root form
    lemmatised_text = " ".join([lemma.lemmatize(word,'v') for word in expanded_text.split()])

    # remove stop words
    stops_free_text = " ".join([word for word in lemmatised_text.split() if word not in stop_words])
    
    return stops_free_text


def load_corpus():
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "lda_topic_modelling", "articles-corpus")
    print(f"corpus-path: {corpus_path}")

    article_paths = [os.path.join(corpus_path,p) for p in os.listdir(corpus_path)]
    print(f"Loading documents!")
    doc_complete = []
    for path in article_paths:
        fp = codecs.open(path,'r','utf-8')
        doc_content = fp.read()
        doc_complete.append(doc_content)  

    sample_size = min(70000, len(doc_complete))
    docs_all = random.sample(doc_complete, sample_size)
    docs = open("docs_wiki.pkl",'wb')
    pickle.dump(docs_all,docs)

    # Use 60000 articles for training.
    docs_train = docs_all[:60000]
    # Use 10000 articles for testing
    docs_test = docs_all[60000:]

    return docs_train, docs_test


def train_model():
    # Load Model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Best for accuracy vs time trade off
    topic_model = BERTopic(embedding_model=embedding_model)
    
    #Load Corpus 
    docs_train, docs_test = load_corpus()
    print(f"docs_train: {len(docs_train)}, docs_test: {len(docs_test)}")
    
    # Clean Documents
    print("Cleaning documents")
    cleaned_docs = [clean(doc) for doc in docs_train]

    print("Fitting the transformer")
    topics, probs = topic_model.fit_transform(cleaned_docs)
    print("Model Trained")

    # Save Model
    topic_model.save("bertopic_model")
    print("Model Saved as 'bertopic_model'")

    # Generate embeddings before writing to a file
    print("Generating embeddings...")
    document_embeddings = embedding_model.encode(cleaned_docs, show_progress_bar=True)

    # Save embeddings correctly
    with open("document_embeddings.pkl", "wb") as f:
        pickle.dump(document_embeddings, f)  # ✅ Use precomputed embeddings
    print("Embeddings saved as 'document_embeddings.pkl'")

def set_topic_labels():
    model = BERTopic.load("bertopic_model")
    topic_info = model.get_topic_info()
    topic_ids = topic_info['Topic']
    topic_names = topic_info['Name']
    topic_representation = topic_info['Representation']
    topic_representative_docs = topic_info['Representative_Docs']
    # for id in topic_ids:
    #     if id != -1: print(f"{id+1}    {topic_names[id+1]}    {topic_representation[id+1]}")
    custom_topic_labels = [
    # Topic -1
    "Can't Determine",
    # Topics 0–199 (200 labels)
    "Movie Reviews",
    "Book Novel",
    "Video Games",
    "Political Figures",
    "TV Actress",
    "French Commune",
    "Marriage Divorce",
    "Air Combat",
    "Web Templates",
    "Car Brands",
    "English Town",
    "Metal Music",
    "British Locomotive",
    "Chemical Compound",
    "Classical Music",
    "Bird Species",
    "Japanese Football",
    "Hockey Draft",
    "Plant Biology",
    "French Region",
    "Hip Hop",
    "Dinosaurs",
    "French Region",
    "Royal Figures",
    "Russian Leaders",
    "German Region",
    "Cell Biology",
    "Animated Series",
    "Infectious Disease",
    "Baseball Pitcher",
    "London Underground",
    "Broadcast Media",
    "Nazi Germany",
    "Space Objects",
    "Art Painting",
    "Tropical Cyclone",
    "Greek Mythology",
    "Secondary School",
    "River Systems",
    "Political Parties",
    "Bollywood Films",
    "Normandy Region",
    "Indian Temple",
    "Swiss Municipal",
    "TV Sitcom",
    "Sarthe Region",
    "Jerusalem Israel",
    "Sports Broadcast",
    "Pakistani Councils",
    "WWE Wrestling",
    "Canadian Regions",
    "NBA Coach",
    "F1 Racing",
    "Middle East",
    "English Football",
    "Logical Propositions",
    "NFL Quarterback",
    "County Census",
    "Periodic Elements",
    "Marine Invertebrates",
    "Electrical Energy",
    "Asian Pop",
    "Italian Football",
    "Island Geography",
    "Math Algebra",
    "Aussie Division",
    "Political Leaders",
    "Glacial River",
    "Mental Disorders",
    "Amphibians",
    "War Allies",
    "Daily News",
    "String Instrument",
    "Military Weapons",
    "Medical Research",
    "Skyscraper Towers",
    "Rodent Burrows",
    "Insect Species",
    "Voice Acting",
    "Civil War",
    "College Campus",
    "Florida Counties",
    "Los Angeles",
    "Celebrity Names",
    "Airline Hub",
    "Disney Animation",
    "Italian Capital",
    "Wiki Notability",
    "Creole Language",
    "Chemical Reaction",
    "River Flow",
    "Military Honors",
    "Japanese Culture",
    "Aussie Politics",
    "Roman Empire",
    "European Cinema",
    "Biblical Literature",
    "Hindu Religion",
    "Computer Hardware",
    "Evolution Biology",
    "South American",
    "Sedimentary Rock",
    "Cricket Test",
    "University Campus",
    "African Capitals",
    "County Metrics",
    "Beatles Music",
    "Pop Music",
    "Art Exhibition",
    "WWE Wrestling",
    "Middle East",
    "Data Encryption",
    "Olympic Games",
    "Persian Poetry",
    "Wildlife Conservation",
    "Geometric Shapes",
    "Greek Regions",
    "Italian Opera",
    "Swiss Canton",
    "French Arrondissements",
    "Citrus Fruit",
    "Korean Athlete",
    "Dutch Football",
    "Eastern Football",
    "Japanese Prefecture",
    "Scottish Regions",
    "Beverage Brands",
    "Web Comments",
    "Water Activities",
    "Rugby Union",
    "Shakespearean Tragedy",
    "Gothic Church",
    "Renewable Energy",
    "Football Manager",
    "Space Shuttle",
    "Catholic Pope",
    "University Students",
    "Fish Species",
    "Military Rank",
    "Food Dish",
    "Geological Time",
    "Digital Archive",
    "Chinese Province",
    "Grammar Tense",
    "Soccer Cup",
    "Int’l Law",
    "Market Economics",
    "German Football",
    "IP Blocking",
    "Constitutional Amendments",
    "Latin Music",
    "Color Shades",
    "Brazilian Football",
    "Fashion Apparel",
    "Baked Goods",
    "Dog Breeds",
    "Retail Stores",
    "US Counties",
    "Middle Eastern",
    "News Anchor",
    "Simpsons Episode",
    "Swedish City",
    "Ski Racing",
    "Galaxy Cluster",
    "Renaissance Art",
    "Japan Football",
    "Anime Manga",
    "Network Protocol",
    "Volcanic Eruption",
    "Irish County",
    "Physics Nobel",
    "Football Stadium",
    "African Leader",
    "Operating System",
    "South American",
    "Wiki Governance",
    "French Region",
    "Olympic Committee",
    "Sexual Topics",
    "Chinese Dynasty",
    "Material States",
    "Marital Status",
    "Programming Code",
    "Highway Route",
    "Medici Dynasty",
    "Criminal Justice",
    "Tennis Slam",
    "Gender Dynamics",
    "Film Awards",
    "Egyptian Mythology",
    "Pakistani Leader",
    "Italian Politics",
    "Reality TV",
    "Legal System",
    "Prime Numbers",
    "Comic Hero",
    "Cardiac Health",
    "US Counties",
    "Web Layout",
    "Neural System",
    
    # Topics 201–400 (200 labels)
    "Singapore MRT",
    "Telephone Code",
    "US Counties",
    "Star Wars",
    "African Football",
    "Islamic Figures",
    "Normandy Region",
    "Mandela Era",
    "Gospel Epistles",
    "Talk Show",
    "Hockey League",
    "Swedish Bandy",
    "Kangaroo Wallaby",
    "Light Waves",
    "Kashmir Dispute",
    "Record Label",
    "Michael Jackson",
    "Ballet Dance",
    "Pregnancy Health",
    "Figure Skating",
    "Urbanization",
    "Texas Counties",
    "Random Stats",
    "Stand-Up Comedy",
    "Software Suite",
    "Spanish Football",
    "Fiction Narrative",
    "River Basins",
    "NFL Football",
    "Calendar Leap",
    "Mathematics",
    "Wood Tools",
    "Zoo Parks",
    "NASCAR Racing",
    "Presidential Wife",
    "Dutch Cities",
    "Venomous Snake",
    "Seoul Festival",
    "Plant Structure",
    "Timpani Music",
    "Textile Weaving",
    "Swimming Race",
    "Irish Parliament",
    "Citation Fields",
    "Baseball Stadium",
    "Korean Leader",
    "Hurricane Season",
    "Poker Game",
    "Pop Band",
    "Currency Coins",
    "Punjabi Language",
    "Argentine Football",
    "Carpet Trade",
    "Drug Addiction",
    "LGBT Issues",
    "Natural Disasters",
    "Air Crash",
    "Polka Parody",
    "Canadian Leader",
    "Comic Heroes",
    "Atlantic Storm",
    "Chess Title",
    "Japanese Athlete",
    "Regional Counties",
    "Olympic Games",
    "Presidential Inauguration",
    "French Football",
    "Jazz Music",
    "Aquitaine Region",
    "Eurovision Song",
    "Light Bulbs",
    "Numeric Template",
    "Floorball Game",
    "Manitoba Assembly",
    "Microbial Taxonomy",
    "Particle Physics",
    "African Languages",
    "Intelligence Agency",
    "Measurement Units",
    "Red Flag",
    "Boxing Champion",
    "Weather Conditions",
    "Wiki Edition",
    "Star Trek",
    "Holiday Celebration",
    "Extinct Mammals",
    "UNESCO Heritage",
    "Supreme Court",
    "Election Vote",
    "Billboard Hit",
    "Chinese Politics",
    "Iowa Counties",
    "Indigenous Art",
    "French Commune",
    "Roller Coaster",
    "New Jersey",
    "Primate Species",
    "Roman Consul",
    "Paralympic Race",
    "Sabbath Holiday",
    "Protestant Church",
    "Seleucid Empire",
    "British Actor",
    "Military Tactics",
    "Egyptian Pharaohs",
    "Fungal Biology",
    "Torchwood TV",
    "Baseball Batter",
    "Astronomy Telescope",
    "Psychoanalysis",
    "Agricultural Crop",
    "Bridge Design",
    "iOS Devices",
    "Fishing Gear",
    "Celestial Orbit",
    "Christmas Song",
    "Sea Creatures",
    "Audio Media",
    "Afghan Province",
    "Illinois Counties",
    "Western Counties",
    "Belgian Regions",
    "Adhesive Materials",
    "Food Chain",
    "Migraine Symptoms",
    "Finnish Regions",
    "Orthodox Church",
    "Singapore Leader",
    "Russian Royalty",
    "Motown Soul",
    "Kennedy Family",
    "Nordic Politics",
    "Phonetics",
    "Tree Species",
    "Historic Castle",
    "Louisiana Parish",
    "Belarus Region",
    "Korean Cities",
    "NHL Standings",
    "Ball Sports",
    "Civil Rights",
    "Japanese Battle",
    "Fuel Oil",
    "Medical Care",
    "Data Query",
    "Phoenician Script",
    "Taxonomy",
    "Country Music",
    "Charity Aid",
    "French Commune",
    "Wiki Permissions",
    "Musical Keys",
    "York County",
    "Beatles Music",
    "Star Constellation",
    "Naval Fleet",
    "Cybersecurity",
    "Alaskan Cities",
    "Tornado Storm",
    "Inventor Patent",
    "Body Joints",
    "Normandy Region",
    "Carolina Counties",
    "Shooting Incident",
    "Turkic Languages",
    "Financial Crime",
    "Spanish Regions",
    "Tropical Cyclone",
    "Supreme Justice",
    "Olympic Medals",
    "Transport Fare",
    "Horse Racing",
    "College Sports",
    "Greek Philosophy",
    "Pakistani Regiment",
    "Tetrapod Fossils",
    "Baptist Ministry",
    "Neolithic Ruins",
    "Armenian Genocide",
    "Mountain Range",
    "Vitamin Toxicity",
    "Charlemagne Dynasty",
    "Time Period",
    "Australopithecines",
    "Chess Codes",
    "Czech Football",
    "Nuclear Accident",
    "Cantonese Culture",
    "Philippine Politics",
    "Hockey Arena",
    "Parasitic Infections",
    "Privy Council",
    "Lizard Species",
    "Feminist Movement",
    "Kentucky Counties",
    "Telecom Provider",
    "Financial Bank",
    "London Boroughs",
    "Japanese Soccer",
    "French Commune",
    
    # Topics 401–600 (200 labels)
    "Gastrointestinal Tract",
    "Space Launch",
    "Celtic Languages",
    "Philippines Cities",
    "WWF Event",
    "Swedish Eurovision",
    "Philosophical Thinkers",
    "Wiki Admin",
    "Oklahoma Region",
    "Wheelchair Sports",
    "Road Traffic",
    "Ontario Highways",
    "Slavic Languages",
    "Presidential Libraries",
    "Software License",
    "Moral Philosophy",
    "Connecticut Towns",
    "Martial Arts",
    "Time Zones",
    "Bicycle Racing",
    "Tectonic Plates",
    "Economic Policy",
    "Mexican Cities",
    "Economics Nobel",
    "Indie Rock",
    "Britney Spears",
    "Footwear",
    "Mortuary Rituals",
    "Photography",
    "Austrian Thought",
    "Welsh Music",
    "Writing Tools",
    "Reagan Speech",
    "Constructed Languages",
    "Kiss Metal",
    "Model Contest",
    "Stellar Types",
    "Satellite Launch",
    "Subway Transit",
    "NZ Politics",
    "Aboriginal Groups",
    "Marvel Comics",
    "Event Timeline",
    "Golf Tournament",
    "Pilgrim History",
    "Southern Towns",
    "Holocaust Memoir",
    "Airport Hubs",
    "Billboard Hits",
    "Bangladesh Politics",
    "Spider Species",
    "Census Stats",
    "Big Cats",
    "Nelly Rap",
    "Horrorcore Rap",
    "Women Athletes",
    "Winter Olympics",
    "Eye Vision",
    "Tennessee Music",
    "Comedy Jokes",
    "Cycling Gear",
    "Muppet Show",
    "Toy Dolls",
    "Computer Input",
    "Harry Potter",
    "Medieval Knights",
    "Alternative Artist",
    "Butterfly Family",
    "Goalball",
    "Mental Health",
    "Lung Cancer",
    "Reality TV",
    "Audio Frequencies",
    "Supercentenarians",
    "Broadway Musical",
    "Swedish Football",
    "Baseball League",
    "Astronomical Observing",
    "Simpsons Characters",
    "Religious Beliefs",
    "Billboard Chart",
    "Convict Fleet",
    "London Circus",
    "Navbox Style",
    "Computing Pioneer",
    "Explosive Devices",
    "Marriage & Sex",
    "Swiss Municipality",
    "Combustion Engine",
    "Turkish Pop",
    "Austrian Football",
    "Backpacker Lodging",
    "Parliament Act",
    "Diabetes Control",
    "Oasis Band",
    "Stonehenge Site",
    "Bahraini Ruler",
    "French Commune",
    "Sleep Disorders",
    "Slade Rock",
    "Groundwater Mining",
    "Museum Collection",
    "Georgia Counties",
    "WP Article",
    "Utah Seal",
    "Azerbaijan Troops",
    "Iranian Football",
    "Anglo-Saxon Kings",
    "Body Parts",
    "Australian Football",
    "WWE Video Game",
    "River Tributary",
    "HK Legislature",
    "Guitar Hero",
    "Japanese Football",
    "Mughal Dynasty",
    "Wiki Interview",
    "Endocrine Glands",
    "Percy Jackson",
    "Beauty Pageant",
    "Wiki Edit",
    "Norse Gods",
    "Sri Lanka Regions",
    "Iranian Politics",
    "Dutch Politics",
    "Cyprus Politics",
    "Tokyo Rail",
    "CFG Sandbox",
    "Notable Deaths",
    "Pornography",
    "Sausage Dog",
    "Historic Plantation",
    "Shrek Animation",
    "Film Production",
    "Ancient Artifacts",
    "Calculus Derivatives",
    "Hindu Temple",
    "Cat Breeds",
    "Samurai Weapons",
    "Windows Server",
    "Measurement Scale",
    "Lock Systems",
    "Avian Dinosaurs",
    "National Anthem",
    "Joseon Dynasty",
    "Fantasy Literature",
    "Massachusetts Cities",
    "Sri Lanka Head",
    "Quantum Mechanics",
    "Normandy Commune",
    "Dormouse Species",
    "French Revolution",
    "Jewish Traditions",
    "Shortwave Radio",
    "Vehicle Queue",
    "Blues Music",
    "Family Relations",
    "Medieval Fortress",
    "Real Estate",
    "Hockey Trophies",
    "Horror Films",
    "River Delta",
    "Theatre Figures",
    "South American Football",
    "Disney Resorts",
    "Hummingbirds",
    "Wiki Permissions",
    "Avatar: Aang",
    "Material Stress",
    "Persistent Pollutants",
    "Photosynthesis Process",
    "James Bond",
    "Independence Signers",
    "Fashion Model",
    "Stream Flow",
    "Mafia Crime",
    "Tropical Cyclones",
    "WWE Divas",
    "Social Media",
    "Portuguese Football",
    "Ballroom Dance",
    "Date Parameter",
    "French Commune",
    "French Festival",
    "Sea Turtles",
    "Taiwan Universities",
    "Hair Styling",
    "Ice Melt",
    "Normandy Commune",
    "Actress Awards",
    "WWE Championship",
    "Anglo Saxons",
    "Historic Castle",
    "Middle East Football",
    "Tarzan Jungle",
    "Football Stadium",
    "Administrative Reform",
    "Shinto Deities",
    "Diplomatic Envoy",
    "Skeletal Anatomy",
    
    # Topics 601–688 (88 labels)
    "Family Guy",
    "Thermodynamic Energy",
    "Teaching & Learning",
    "Wiki Archive",
    "Metal Corrosion",
    "Operetta Theatre",
    "WWE Wrestling",
    "Telenovela Soap",
    "Hokkaido Prefecture",
    "Coat of Arms",
    "Caribbean Parish",
    "Australian Suburb",
    "Global Summit",
    "Nepal Region",
    "Wiki Protection",
    "Green Day",
    "Italian Poets",
    "Waste Recycling",
    "River Fork",
    "Rail Signal",
    "Config Settings",
    "Stamp Philately",
    "Normandy Commune",
    "Urawa Football",
    "Reggae Music",
    "Loan & Debt",
    "Brazil Census",
    "Paralympic Cycling",
    "Senate Contest",
    "Hercules Legend",
    "Web Search",
    "Marxism Ideas",
    "Martial Arts",
    "Evergrande China",
    "Hurricane Season",
    "Sapphire Mineral",
    "Graph Theory",
    "Rock Band",
    "Rail Station",
    "Typography",
    "Carpentry Crafts",
    "Colonialism",
    "Picardie Commune",
    "Earthquake Shake",
    "Calendar Hours",
    "Music Prize",
    "Electrical Software",
    "Telecom Cards",
    "Space Probes",
    "Disney Cartoons",
    "Tasman Coast",
    "Danish Football",
    "UK Counties",
    "Chicago Loop",
    "Gregorian Calendar",
    "Wiki Categories",
    "Calais Nord",
    "Indonesian Islands",
    "Fruit Varieties",
    "Polar Oceans",
    "Respiratory System",
    "Tort Law",
    "Women Tournament",
    "Hermitage Museum",
    "London Boroughs",
    "Lat Lon Data",
    "Volleyball Medals",
    "Political Families",
    "Chocolate Beverage",
    "Diplomatic Envoy",
    "Cyrillic Letter",
    "Bundesliga Football",
    "Romanian Football",
    "Lab Glassware",
    "Rope Ladder",
    "Brassica Veg",
    "Indoor Lacrosse",
    "Karachi Beach",
    "Indie Bloc",
    "Viking Metal",
    "Czech Regions",
    "College Football",
    "Christian Faith",
    "Payment Audit",
    "Romanian Regions",
    "Animal Packs",
    "Belize Towns",
    "Tibetan Buddhism"
]


    # Set Custom Labels
    model.set_topic_labels(custom_topic_labels)

    # Get custom labels
    custom_labels = model.get_topic_info()['CustomName']
    print(f"Custom labels are : \n{custom_labels}")

    model.save("bertopic_model_custom_labels")
    print("custome labeled model is saved as 'bertopic_model_custom_labels'")

def get_theme(text):
    # load the trained model
    current_path = os.path.dirname(__file__)
    bert_path = os.path.join(current_path, "bertopic_model_custom_labels")
    topic_model = BERTopic.load(bert_path)

    # load the custom labels
    custom_labels  = topic_model.get_topic_info()['CustomName']

    # predict the top 3 topics
    top_topics, top_probs = topic_model.find_topics(text, top_n = 3)

    top_names = []

    for i, topic_id in enumerate(top_topics):
        topic_words = topic_model.get_topic(topic_id)  # Get words for topic
        topic_name = custom_labels[topic_id+1]  # Get custom name if available
        # add to the result list
        top_names.append(topic_name)
    return [top_names, top_probs]

if __name__ == "__main__":
    test_text = "books author publish published"
    print(get_theme(test_text))
    pass