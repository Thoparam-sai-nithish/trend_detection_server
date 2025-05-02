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

def load_corpus(train_size):
    # Path to the 'data' folder where the pickled files are stored
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    print(f"Data directory: {data_dir}")
    
    # Full paths to the cleaned pickle files
    train_path = os.path.join(data_dir, f"docs_train_{train_size}.pkl")
    test_path = os.path.join(data_dir, f"docs_test_{train_size}.pkl")

    # Load the cleaned documents
    with open(train_path, "rb") as f:
        docs_train = pickle.load(f)
    with open(test_path, "rb") as f:
        docs_test = pickle.load(f)

    print(f"Loaded {len(docs_train)} training documents and {len(docs_test)} testing documents.")

    return docs_train, docs_test



def train_model():

    # Load Model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # Best for accuracy vs time trade off
    topic_model = BERTopic(embedding_model=embedding_model)
    
    #Load Corpus 
    train_size = 90
    docs_train, docs_test = load_corpus(train_size)
    print(f"docs_train: {len(docs_train)}, docs_test: {len(docs_test)}")

    print("Fitting the transformer")
    topics, probs = topic_model.fit_transform(docs_train)
    print("Model Trained")

    # Save Model
    topic_model.save(f"{train_size}_bertopic_model")
    print(f"Model Saved as '{train_size}_bertopic_model'")

    # Generate embeddings before writing to a file
    print("Generating embeddings...")
    document_embeddings = embedding_model.encode(docs_train, show_progress_bar=True)

    # Save embeddings correctly
    with open(f"{train_size}_document_embeddings.pkl", "wb") as f:
        pickle.dump(document_embeddings, f)  # ✅ Use precomputed embeddings
    print(f"Embeddings saved as '{train_size}_document_embeddings.pkl'")

    # Save the BERTopic summary
    print("Saving topic summary...")
    with open(f"{train_size}_bertopic_summary.txt", "w", encoding="utf-8") as f:
        # Get the topics information
        topic_info = topic_model.get_topic_info()

        # Iterate through the topic info and write it to the file
        for i, row in topic_info.iterrows():
            # Get the topic ID and number of documents
            topic_id = row['Topic']
            num_docs = row['Count']
            if topic_id == -1:
                f.write(f"Outlier Topic: {topic_id} (Documents: {num_docs})\n")
            else:
                words = topic_model.get_topic(topic_id)  # Get top words for the topic
                words_str = ", ".join([w for w, _ in words])
                f.write(f"Topic {topic_id}: {words_str} (Documents: {num_docs})\n")
    
    print(f"✅ BERTopic summary saved to '{train_size}_bertopic_summary.txt'")

def set_topic_labels():
    model = BERTopic.load("70_bertopic_model")
    topic_info = model.get_topic_info()
    topic_ids = topic_info['Topic']
    topic_names = topic_info['Name']
    topic_representation = topic_info['Representation']
    topic_representative_docs = topic_info['Representative_Docs']
    # for id in topic_ids:
    #     if id != -1: print(f"{id+1}    {topic_names[id+1]}    {topic_representation[id+1]}")
    custom_topic_labels = [
        "common_words",    #topic_-1
        "movie_drama",    #topic_0
        "game_console",    #topic_1
        "book_novel",    #topic_2
        "actress_television",    #topic_3
        "calais_france",    #topic_4
        "borough_town",    #topic_5
        "king_duke",    #topic_6
        "band_album",    #topic_7
        "marry_divorce",    #topic_8
        "alphabet_language",    #topic_9
        "picardie_france",    #topic_10
        "paint_art",    #topic_11
        "str_template",    #topic_12
        "flower_plant",    #topic_13
        "university_college",    #topic_14
        "aircraft_fighter",    #topic_15
        "ford_motor",    #topic_16
        "rail_locomotive",    #topic_17
        "chemical_oxidation",    #topic_18
        "hockey_nhl",    #topic_19
        "dinosaur_cretaceous",    #topic_20
        "japan_football",    #topic_21
        "bird_feather",    #topic_22
        "orchestra_composer",    #topic_23
        "governor_senate",    #topic_24
        "gironde_aquitaine",    #topic_25
        "mythology_greek",    #topic_26
        "hip_hop",    #topic_27
        "church_catholic",    #topic_28
        "batman_comics",    #topic_29
        "mineral_rock",    #topic_30
        "song_single",    #topic_31
        "storm_hurricane",    #topic_32
        "airport_airlines",    #topic_33
        "skate_olympics",    #topic_34
        "germany_rhine",    #topic_35
        "russian_soviet",    #topic_36
        "sauce_dish",    #topic_37
        "baseball_league",    #topic_38
        "jerusalem_israel",    #topic_39
        "energy_electricity",    #topic_40
        "station_railway",    #topic_41
        "nfl_draft",    #topic_42
        "disney_walt",    #topic_43
        "calvados_normandie",    #topic_44
        "korean_pop",    #topic_45
        "wwe_event",    #topic_46
        "murder_shoot",    #topic_47
        "wikipedia_article",    #topic_48
        "switzerland_canton",    #topic_49
        "instruction_memory",    #topic_50
        "islands_archipelago",    #topic_51
        "bible_testament",    #topic_52
        "basketball_nba",    #topic_53
        "river_glacier",    #topic_54
        "fish_shark",    #topic_55
        "urdu_murad",    #topic_56
        "subspecies_tiger",    #topic_57
        "district_india",    #topic_58
        "espn_sportscaster",    #topic_59
        "sarthe_loire",    #topic_60
        "labour_conservative",    #topic_61
        "blood_heart",    #topic_62
        "hindu_hinduism",    #topic_63
        "voice_actor",    #topic_64
        "division_queensland",    #topic_65
        "italy_province",    #topic_66
        "tehsil_councils",    #topic_67
        "college_university",    #topic_68
        "tallest_tower",    #topic_69
        "gun_rifle",    #topic_70
        "canada_ontario",    #topic_71
        "beetle_insects",    #topic_72
        "element_periodic",    #topic_73
        "drug_addiction",    #topic_74
        "radio_broadcast",    #topic_75
        "cartoon_nickelodeon",    #topic_76
        "alabama_county",    #topic_77
        "iran_kurdish",    #topic_78
        "italy_football",    #topic_79
        "crab_snail",    #topic_80
        "frog_toad",    #topic_81
        "television_show",    #topic_82
        "singer_songwriter",    #topic_83
        "nobel_physiology",    #topic_84
        "wear_clothe",    #topic_85
        "archive_discussion",    #topic_86
        "wrestle_wrestler",    #topic_87
        "suicide_child",    #topic_88
        "newspaper_magazine",    #topic_89
        "name_female",    #topic_90
        "germany_hitler",    #topic_91
        "dominican_chile",    #topic_92
        "california_san",    #topic_93
        "native_maya",    #topic_94
        "instrument_string",    #topic_95
        "calendar_year",    #topic_96
        "formula_race",    #topic_97
        "geometry_angle",    #topic_98
        "pope_catholic",    #topic_99
        "japanese_japan",    #topic_100
        "sexual_identity_101",    #topic_101
        "river_system_102",    #topic_102
        "religious_belief_103",    #topic_103
        "wisconsin_county_104",    #topic_104
        "galaxy_cluster_105",    #topic_105
        "civil_war_106",    #topic_106
        "islamic_belief_107",    #topic_107
        "evolution_theory_108",    #topic_108
        "joseon_dynasty_109",    #topic_109
        "roman_empire_110",    #topic_110
        "hockey_league_111",    #topic_111
        "dutch_language_112",    #topic_112
        "florida_county_113",    #topic_113
        "baking_desserts_114",    #topic_114
        "constitutional_amendment_115",    #topic_115
        "gothic_architecture_116",    #topic_116
        "actor_roles_117",    #topic_117
        "national_flag_118",    #topic_118
        "color_pigments_119",    #topic_119
        "dutch_football_120",    #topic_120
        "particle_physics_121",    #topic_121
        "african_football_122",    #topic_122
        "shakespeare_plays_123",    #topic_123
        "japanese_prefecture_124",    #topic_124
        "extreme_sports_125",    #topic_125
        "middle_eastern_politics_126",    #topic_126
        "olympic_athletes_127",    #topic_127
        "measurement_units_128",    #topic_128
        "french_arrondissement_129",    #topic_129
        "connecticut_town_130",    #topic_130
        "japanese_football_131",    #topic_131
        "woodworking_tools_132",    #topic_132
        "missouri_county_133",    #topic_133
        "cryptography_algorithms_134",    #topic_134
        "mental_health_135",    #topic_135
        "geological_periods_136",    #topic_136
        "ancient_egypt_137",    #topic_137
        "greek_islands_138",    #topic_138
        "korean_athletes_139",    #topic_139
        "wikipedia_comments_140",    #topic_140
        "human_rights_141",    #topic_141
        "programming_languages_142",    #topic_142
        "linux_systems_143",    #topic_143
        "german_football_144",    #topic_144
        "swiss_municipalities_145",    #topic_145
        "weather_conditions_146",    #topic_146
        "scottish_football_147",    #topic_147
        "anime_series_148",    #topic_148
        "software_licenses_149",    #topic_149
        "football_cup_150",    #topic_150
        "latin_music_151",    #topic_151
        "scottish_towns_152",    #topic_152
        "opera_composers_153",    #topic_153
        "kashmir_dispute_154",    #topic_154
        "english_football_155",    #topic_155
        "dog_breeds_156",    #topic_156
        "boxing_champions_157",    #topic_157
        "rugby_league_158",    #topic_158
        "military_medals_159",    #topic_159
        "spanish_football_160",    #topic_160
        "swedish_sports_161",    #topic_161
        "highway_systems_162",    #topic_162
        "australian_labor_163",    #topic_163
        "israeli_politics_164",    #topic_164
        "computer_networks_165",    #topic_165
        "sheffield_football_166",    #topic_166
        "chemical_reactions_167",    #topic_167
        "chinese_regions_168",    #topic_168
        "victorian_australia_169",    #topic_169
        "school_education_170",    #topic_170
        "matrix_theory_171",    #topic_171
        "brazil_football_172",    #topic_172
        "world_war_ii_173",    #topic_173
        "photography_gear_174",    #topic_174
        "olympic_organization_175",    #topic_175
        "volcanic_eruption_176",    #topic_176
        "mountain_ranges_177",    #topic_177
        "russian_football_178",    #topic_178
        "bacteria_species_179",    #topic_179
        "space_missions_180",    #topic_180
        "academy_awards_181",    #topic_181
        "election_process_182",    #topic_182
        "market_economics_183",    #topic_183
        "south_american_football_184",    #topic_184
        "neptune_moon_185",    #topic_185
        "michael_jackson_186",    #topic_186
        "nascar_racing_187",    #topic_187
        "mayenne_region_188",    #topic_188
        "simpsons_episodes_189",    #topic_189
        "philosophy_wisdom_190",    #topic_190
        "indian_politics_191",    #topic_191
        "immune_system_192",    #topic_192
        "ballet_dance_193",    #topic_193
        "reagan_bush_194",    #topic_194
        "medical_care_195",    #topic_195
        "singapore_transit_196",    #topic_196
        "mobile_technology_197",    #topic_197
        "anglo_saxons_198",    #topic_198
        "economic_poverty_199",    #topic_199
        "currency_exchange_200",    #topic_200
        "mathematician_theory",    #topic_201
        "nervous_system",    #topic_202
        "algae_classification",    #topic_203
        "holiday_sabbath",    #topic_204
        "normandie_france",    #topic_205
        "paralympic_swimmer",    #topic_206
        "french_football",    #topic_207
        "eurovision_contest",    #topic_208
        "azerbaijan_karabakh",    #topic_209
        "reagan_secretary",    #topic_210
        "physics_nobel",    #topic_211
        "restaurant_chain",    #topic_212
        "wikipedia_edition",    #topic_213
        "italian_rome",    #topic_214
        "dakota_county",    #topic_215
        "irish_county",    #topic_216
        "jedi_sith",    #topic_217
        "senate_party",    #topic_218
        "italian_berneri",    #topic_219
        "digestive_system",    #topic_220
        "lgbt_identity",    #topic_221
        "egyptian_mythology",    #topic_222
        "nigeria_capital",    #topic_223
        "tv_host",    #topic_224
        "belarus_russia",    #topic_225
        "railway_transport",    #topic_226
        "earth_orbit",    #topic_227
        "pakistan_bhutto",    #topic_228
        "cancer_treatment",    #topic_229
        "fruit_vegetable",    #topic_230
        "baseball_stadium",    #topic_231
        "secretary_cabinet",    #topic_232
        "marsupial_animals",    #topic_233
        "crop_farming",    #topic_234
        "china_communist",    #topic_235
        "genetics_dna",    #topic_236
        "cricket_match",    #topic_237
        "snake_species",    #topic_238
        "rock_band",    #topic_239
        "mughal_dynasty",    #topic_240
        "coordinate_system",    #topic_241
        "germany_treaty",    #topic_242
        "historical_events",    #topic_243
        "weird_al",    #topic_244
        "chess_grandmaster",    #topic_245
        "hurricane_storm",    #topic_246
        "investment_business",    #topic_247
        "fungi_mushroom",    #topic_248
        "athlete_sport",    #topic_249
        "supermarket_store",    #topic_250
        "homo_fossil",    #topic_251
        "sweden_politics",    #topic_252
        "wool_fabric",    #topic_253
        "chernobyl_nuclear",    #topic_254
        "mgm_entertainment",    #topic_255
        "normandie_regions",    #topic_256
        "oklahoma_county",    #topic_257
        "asteroid_belt",    #topic_258
        "logic_inference",    #topic_259
        "hurricane_button",    #topic_260
        "doctor_companion",    #topic_261
        "korean_politics",    #topic_262
        "ballroom_dance",    #topic_263
        "spain_catalonia",    #topic_264
        "texas_county",    #topic_265
        "dutch_writer",    #topic_266
        "legal_court",    #topic_267
        "kenya_politician",    #topic_268
        "wimbledon_tennis",    #topic_269
        "block_address",    #topic_270
        "nbc_news",    #topic_271
        "holiday_valentine",    #topic_272
        "probability_statistics",    #topic_273
        "presidential_inauguration",    #topic_274
        "sweden_football",    #topic_275
        "css_style",    #topic_276
        "freud_psychology",    #topic_277
        "jersey_county",    #topic_278
        "bantu_languages",    #topic_279
        "supreme_court",    #topic_280
        "engine_rotor",    #topic_281
        "cite_reference",    #topic_282
        "privy_council",    #topic_283
        "carolina_county",    #topic_284
        "municipality_sweden",    #topic_285
        "irish_northern",    #topic_286
        "baseball_uniform",    #topic_287
        "poker_game",    #topic_288
        "picardie_france",    #topic_289
        "inventor_patent",    #topic_290
        "french_paris",    #topic_291
        "olympics_olympiad",    #topic_292
        "wrestling_championship",    #topic_293
        "museum_exhibit",    #topic_294
        "plant_growth",    #topic_295
        "musical_notes",    #topic_296
        "astronomy_telescope",    #topic_297
        "cell_respiration",    #topic_298
        "chromosome_reproduction",    #topic_299
        "tropical_storm",    #topic_300
        "finland_ostrobothnia",    #topic_301
        "lizard_gecko",    #topic_302
        "illinois_ohio",    #topic_303
        "interview_user",    #topic_304
        "punjabi_dialect",    #topic_305
        "consulship_julian",    #topic_306
        "gironde_aquitaine",    #topic_307
        "economics_nobel",    #topic_308
        "athlete_paralympics",    #topic_309
        "iowa_cedar",    #topic_310
        "alaska_nome",    #topic_311
        "aboriginal_art",    #topic_312
        "golf_pga",    #topic_313
        "cities_density",    #topic_314
        "coaster_roller",    #topic_315
        "trek_enterprise",    #topic_316
        "philippines_cebu",    #topic_317
        "asian_martial",    #topic_318
        "uterus_hormone",    #topic_319
        "fraud_robbery",    #topic_320
        "permissions_request",    #topic_321
        "heer_bhattacharya",    #topic_322
        "thermodynamics_thermodynamic",    #topic_323
        "spiders_tarantula",    #topic_324
        "pd_pl",    #topic_325
        "circuit_prix",    #topic_326
        "festival_seoul",    #topic_327
        "rivers_drain",    #topic_328
        "slavic_languages",    #topic_329
        "cd_disc",    #topic_330
        "bosnia_kosovo",    #topic_331
        "library_libraries",    #topic_332
        "residence_palace",    #topic_333
        "afghanistan_province",    #topic_334
        "rank_army",    #topic_335
        "story_fiction",    #topic_336
        "peat_forest",    #topic_337
        "lock_abloy",    #topic_338
        "joint_muscle",    #topic_339
        "aisne_picardie",    #topic_340
        "queue_walker",    #topic_341
        "hockey_trophy",    #topic_342
        "wonder_stevie",    #topic_343
        "court_justice",    #topic_344
        "oil_water",    #topic_345
        "wikipedia_edit",    #topic_346
        "mandela_apartheid",    #topic_347
        "loch_nairn",    #topic_348
        "bridge_manhattan",    #topic_349
        "arena_hockey",    #topic_350
        "deaths_notable",    #topic_351
        "ship_convict",    #topic_352
        "rayon_armenian",    #topic_353
        "census_bureau",    #topic_354
        "request_checkusership",    #topic_355
        "philippines_filipino",    #topic_356
        "ski_skier",    #topic_357
        "counties_metropolitan",    #topic_358
        "louisiana_parish",    #topic_359
        "constellation_orion",    #topic_360
        "bone_pelvis",    #topic_361
        "russia_romanov",    #topic_362
        "xtc_band",    #topic_363
        "liquid_solid",    #topic_364
        "ski_cross",    #topic_365
        "telephone_dial",    #topic_366
        "goalball_paralympics",    #topic_367
        "turkey_turkish",    #topic_368
        "funeral_coffin",    #topic_369
        "czech_bulgaria",    #topic_370
        "austrian_austria",    #topic_371
        "vitamin_overweight",    #topic_372
        "sex_intercourse",    #topic_373
        "heritage_sit",    #topic_374
        "anthem_zambia",    #topic_375
        "loudspeaker_amplifier",    #topic_376
        "usher_chart",    #topic_377
        "radiation_wavelength",    #topic_378
        "aisne_picardie",    #topic_379
        "earthquake_magnitude",    #topic_380
        "diary_holocaust",    #topic_381
        "temple_hindu",    #topic_382
        "jazz_trumpeter",    #topic_383
        "horse_race",    #topic_384
        "georgia_gwinnett",    #topic_385
        "convention_republican",    #topic_386
        "cricket_cricketer",    #topic_387
        "zoo_zoos",    #topic_388
        "parasite_parasites",    #topic_389
        "glass_glue",    #topic_390
        "drink_pepsi",    #topic_391
        "seal_list",    #topic_392
        "mothball_pollutants",    #topic_393
        "protour_tour",    #topic_394
        "maida_raine",    #topic_395
        "request_oversightship",    #topic_396
        "bin_khalifa",    #topic_397
        "calvados_normandie",    #topic_398
        "mccartney_beatles",    #topic_399
        "algorithm_graph",    #topic_400
        "iran_islamic",    #topic_401
        "mobile_provider",    #topic_402
        "winter_olympics",    #topic_403
        "football_players",    #topic_404
        "canadian_highway",    #topic_405
        "rockabilly_band",    #topic_406
        "postage_stamp",    #topic_407
        "presidential_libraries",    #topic_408
        "nhl_standings",    #topic_409
        "time_zone",    #topic_410
        "london_boroughs",    #topic_411
        "climate_sustainability",    #topic_412
        "oil_fuel",    #topic_413
        "css_styling",    #topic_414
        "wwe_video_game",    #topic_415
        "unlucky_numbers",    #topic_416
        "subway_service",    #topic_417
        "canadian_minister",    #topic_418
        "australia_explorer",    #topic_419
        "antivirus_malware",    #topic_420
        "stock_exchange",    #topic_421
        "muppet_characters",    #topic_422
        "football_quarterback",    #topic_423
        "christmas_carol",    #topic_424
        "number_system",    #topic_425
        "oldest_person",    #topic_426
        "irish_football",    #topic_427
        "radio_signal",    #topic_428
        "social_network",    #topic_429
        "australian_rivers",    #topic_430
        "kentucky_counties",    #topic_431
        "melbourne_railway",    #topic_432
        "devonian_fossils",    #topic_433
        "adminship_wikipedia",    #topic_434
        "ramsar_lakes",    #topic_435
        "military_strategy",    #topic_436
        "broadway_theatre",    #topic_437
        "singapore_prime",    #topic_438
        "music_record_label",    #topic_439
        "stonehenge_site",    #topic_440
        "bridge_traffic",    #topic_441
        "satellite_orbit",    #topic_442
        "earthquake_seismic",    #topic_443
        "deletion_policy",    #topic_444
        "canadian_party",    #topic_445
        "japanese_football",    #topic_446
        "irc_server",    #topic_447
        "school_students",    #topic_448
        "heavy_metal_band",    #topic_449
        "oasis_band",    #topic_450
        "disneyland_park",    #topic_451
        "castle_harley",    #topic_452
        "billiards_ball",    #topic_453
        "welsh_instrumentalist",    #topic_454
        "intellectual_property",    #topic_455
        "britney_spear",    #topic_456
        "korean_cities",    #topic_457
        "hong_kong_actors",    #topic_458
        "romania_carpathians",    #topic_459
        "light_bulb",    #topic_460
        "telenovela_televisa",    #topic_461
        "super_bowl",    #topic_462
        "delaware_castle",    #topic_463
        "music_award",    #topic_464
        "cycle_model",    #topic_465
        "hercules_legendary",    #topic_466
        "clive_scofield",    #topic_467
        "cosplay_costume",    #topic_468
        "komodo_park",    #topic_469
        "tarzan_jungle",    #topic_470
        "calculus_derivative",    #topic_471
        "tornado_outbreak",    #topic_472
        "railway_station",    #topic_473
        "beauty_pageant",    #topic_474
        "king_arthur",    #topic_475
        "chihuahua_mexico",    #topic_476
        "japan_prefectures",    #topic_477
        "vinyl_recording",    #topic_478
        "student_education",    #topic_479
        "avatar_airbender",    #topic_480
        "cartoon_characters",    #topic_481
        "windows_microsoft",    #topic_482
        "printer_inkjet",    #topic_483
        "primate_monkey",    #topic_484
        "cargo_ship",    #topic_485
        "cell_organelles",    #topic_486
        "cat_breeds",    #topic_487
        "kansas_staters",    #topic_488
        "prime_number",    #topic_489
        "kantner_guitar",    #topic_490
        "protest_riot",    #topic_491
        "survivor_tribe",    #topic_492
        "aviation_intelligence",    #topic_493
        "nelly_rapper",    #topic_494
        "slade_band",    #topic_495
        "leap_decade",    #topic_496
        "ethics_moral",    #topic_497
        "belgium_senate",    #topic_498
        "kiss_metal",    #topic_499
        "cyclist_paralympics",    #topic_500
        "nepal_development",    #topic_501
        "dormouse_rodent",    #topic_502
        "epa_homeland",    #topic_503
        "request_adminship",    #topic_504
        "australia_football",    #topic_505
        "beach_karachi",    #topic_506
        "spacecraft_probe",    #topic_507
        "bbc_show",    #topic_508
        "pakistan_infantry",    #topic_509
        "constituency_hong",    #topic_510
        "sweden_uppsala",    #topic_511
        "mar_greenhouse",    #topic_512
        "nazi_hitler",    #topic_513
        "harry_potter",    #topic_514
        "chemistry_synthesis",    #topic_515
        "hero_guitar",    #topic_516
        "ball_kick",    #topic_517
        "tennis_itf",    #topic_518
        "band_gutierrez",    #topic_519
        "virus_mimivirus",    #topic_520
        "mendrisio_switzerland",    #topic_521
        "ecuador_quito",    #topic_522
        "sword_tanto",    #topic_523
        "socialism_capitalism",    #topic_524
        "autism_spectrum",    #topic_525
        "baffin_nunavut",    #topic_526
        "rehearsal_performance",    #topic_527
        "percy_olympians",    #topic_528
        "petrarch_zeichen",    #topic_529
        "fashion_laurent",    #topic_530
        "calais_gohelle",    #topic_531
        "brazil_ibge",    #topic_532
        "degree_bachelor",    #topic_533
        "urawa_diamonds",    #topic_534
        "page_protect",    #topic_535
        "ferns_bryophytes",    #topic_536
        "portugal_benfica",    #topic_537
        "wales_south",    #topic_538
        "permissions_bureaucrat",    #topic_539
        "mayflower_carver",    #topic_540
        "urban_plan",    #topic_541
        "butterfly_admiral",    #topic_542
        "battery_rechargeable",    #topic_543
        "fat_sugar",    #topic_544
        "dive_underwater",    #topic_545
        "armenian_oni",    #topic_546
        "tokyo_japan",    #topic_547
        "blue_harmonica",    #topic_548
        "train_ballast",    #topic_549
        "button_hurricane",    #topic_550
        "zealand_prime",    #topic_551
        "bridge_thames",    #topic_552
        "indie_rock",    #topic_553
        "brewery_beer",    #topic_554
        "chart_billboard",    #topic_555
        "holly_jennings",    #topic_556
        "metal_punk",    #topic_557
        "milkweed_asclepiadoideae",    #topic_558
        "castle_fortification",    #topic_559
        "commentry_french",    #topic_560
        "fish_bait",    #topic_561
        "persian_khomeini",    #topic_562
        "model_tookes",    #topic_563
        "illinois_fermilab",    #topic_564
        "barbados_parish",    #topic_565
        "fruit_plantain",    #topic_566
        "bambi_godzilla",    #topic_567
        "socratic_empedocles",    #topic_568
        "hummingbird_moist",    #topic_569
        "cheese_milk",    #topic_570
        "nara_acquire",    #topic_571
        "declaration_party",    #topic_572
        "sociologist_sociology",    #topic_573
        "threaten_conservation",    #topic_574
        "nirvana_grunge",    #topic_575
        "exercise_muscle",    #topic_576
        "eye_pupil",    #topic_577
        "townshend_act",    #topic_578
        "eagleton_barr",    #topic_579
        "bangkok_palace",    #topic_580
        "iso_cod",    #topic_581
        "heflin_illinois",    #topic_582
        "iwgp_wrestle",    #topic_583
        "babbage_toffler",    #topic_584
        "shinto_kami",    #topic_585
        "redwall_jacques",    #topic_586
        "space_spaceflight",    #topic_587
        "andrews_sink",    #topic_588
        "boccia_paralympics",    #topic_589
        "hiddencat_categorytoc",    #topic_590
        "huntingdon_animal",    #topic_591
        "diabetes_insulin",    #topic_592
        "ebert_evaluation",    #topic_593
        "nicotine_tobacco",    #topic_594
        "comedian_carlin",    #topic_595
        "battle_cornwallis",    #topic_596
        "marathon_run",    #topic_597
        "exhibition_moscow",    #topic_598
        "judaism_rabbis",    #topic_599
        "presidential_campaign",    #topic_600
        "chemical_kinetics",    #topic_601
        "musical_instrument",    #topic_602
        "russian_communism",    #topic_603
        "angola_province",    #topic_604
        "sleep_disorder",    #topic_605
        "basketball_nba",    #topic_606
        "astronomer_gehrels",    #topic_607
        "bond_royale",    #topic_608
        "cultural_history",    #topic_609
]

    
    print(f"No.of Topics : {len(custom_topic_labels)}")
    # Set Custom Labels
    model.set_topic_labels(custom_topic_labels)

    # Get custom labels
    custom_labels = model.get_topic_info()['CustomName']
    print(f"Custom labels are : \n{custom_labels}")

    model.save("bertopic_model_custom_labels")
    print("custome labeled model is saved as 'bertopic_model_custom_labels'")

def get_theme(text):
    # load the trained model
    train_size = 70
    current_path = os.path.dirname(__file__)
    bert_path = os.path.join(current_path, f"bertopic_model_custom_labels")
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
    # train_model()
    article = "Music is the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content. Music is generally agreed to be a cultural universal that is present in all human societies. Definitions of music vary widely in substance and approach."
    print(article, "\n")

    print("Theme -> ",get_theme(article))
    pass
