"""Evaluation datasets (golden sets). Reference answers for medical QA are
paraphrased from MedlinePlus / WHO pages in the knowledge corpus; reference
patient summaries come from the ``Summary`` column of records.xlsx where
available, otherwise were written from the source PDFs."""

# (actor_id, actor_role, query, expected_tools, expected_specialty, expected_patient_name)
PLANNER_CASES = [
    ("P001", "patient", "My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
     "Also, can you summarize latest treatment methods?",
     {"identify_patient", "get_patient_history", "book_appointment", "medical_info_search"}, "Nephrology", "Suresh Negi"),
    ("P001", "patient", "Book a cardiologist for my dad next Monday morning",
     {"identify_patient", "get_patient_history", "book_appointment"}, "Cardiology", "Suresh Negi"),
    ("P001", "patient", "What medications is my father currently taking? Summarize his history.",
     {"identify_patient", "get_patient_history"}, None, "Suresh Negi"),
    ("P001", "patient", "Show my father's upcoming appointments", {"identify_patient", "list_appointments"}, None, "Suresh Negi"),
    ("P004", "patient", "I have had a dry cough for a week, can I see a general physician tomorrow?",
     {"identify_patient", "get_patient_history", "book_appointment"}, "General Medicine", "Anjali Mehra"),
    ("P004", "patient", "What are the symptoms of migraine?", {"medical_info_search"}, None, None),
    ("P005", "patient", "What is the best diet for type 2 diabetes?", {"medical_info_search"}, None, None),
    ("P005", "patient", "Cancel my endocrinology appointment", {"identify_patient", "cancel_appointment"}, None, "David Thompson"),
    ("P003", "patient", "Please reschedule my cardiology appointment to Friday afternoon",
     {"identify_patient", "reschedule_appointment"}, None, "Ramesh Kulkarni"),
    ("attendant-01", "attendant", "Summarize the medical history of Rebeca Nagle",
     {"identify_patient", "get_patient_history"}, None, "Rebeca Nagle"),
    ("attendant-01", "attendant", "Update the phone number of David Thompson to +91-99999-00000",
     {"identify_patient", "update_patient_record"}, None, "David Thompson"),
    ("attendant-01", "attendant", "Add a note for Ramesh Kulkarni: BP 150/95 today, started amlodipine 5mg OD.",
     {"identify_patient", "add_clinical_note"}, None, "Ramesh Kulkarni"),
    ("attendant-01", "attendant", "Book an endocrinologist for David Thompson and tell me about diabetic kidney disease",
     {"identify_patient", "get_patient_history", "book_appointment", "medical_info_search"}, "Endocrinology", "David Thompson"),
    ("P002", "patient", "I need a gynecologist appointment this week", {"identify_patient", "get_patient_history", "book_appointment"},
     "Obstetrics & Gynecology", "Rebeca Nagle"),
    ("P004", "patient", "Update my address to 14 MG Road, Pune", {"identify_patient"}, None, "Anjali Mehra"),  # role guard strips update
]

# patient name -> reference summary
SUMMARY_CASES = {
    "Ramesh Kulkarni": "Patient presents for routine checkup with a history of hypertension. Vitals are stable. Continue "
                       "current medication (Telmisartan 40mg OD) and recommend lifestyle modifications. Routine labs "
                       "ordered and return in 6 months.",
    "Anjali Mehra": "Patient presents with 5-day history of dry cough and mild fever, diagnosed with Upper Respiratory "
                    "Infection (J06.9), and advised symptomatic management with antihistamines and fluids, rest, and "
                    "follow-up in 5 days.",
    "David Thompson": "Patient presents for follow-up of Type 2 Diabetes. Reports increased thirst and urination. Diagnosis: "
                      "Type 2 Diabetes Mellitus (E11.9). Plan: Increase metformin dosage to 1000mg BID, order HbA1c and "
                      "Lipid Profile, schedule nutritionist follow-up, return in 3 months.",
    "Rebeca Nagle": "Female patient with PCOS and migraines on magnesium and vitamin B2. Well-woman exam in September 2022 "
                    "with Pap smear and flu shot; screening labs showed high ALT, AST, glucose and triglycerides and low "
                    "HDL. Later diagnosed with costochondritis (M94.0) treated with ibuprofen, stretching and rest.",
    "Suresh Negi": "70-year-old man with chronic kidney disease stage 3b due to type 2 diabetes and hypertension, with "
                   "albuminuria, high potassium and anemia. On losartan, empagliflozin, reduced-dose metformin and "
                   "atorvastatin. Referred to nephrology and a dietitian; allergic to sulfa drugs.",
}

# (question, reference answer, key terms that a correct answer should mention)
MEDICAL_QA_CASES = [
    ("What are the most common causes of chronic kidney disease?",
     "Diabetes and high blood pressure are the most common causes of chronic kidney disease.",
     ["diabetes", "blood pressure"]),
    ("How is kidney failure treated?",
     "Kidney failure is treated with dialysis or a kidney transplant.",
     ["dialysis", "transplant"]),
    ("What are the symptoms of type 2 diabetes?",
     "Symptoms include increased thirst and urination, increased hunger, fatigue, blurred vision, numbness or tingling "
     "in the feet or hands, sores that do not heal and unexplained weight loss; many people have no symptoms.",
     ["thirst", "urinat", "tired|fatigue", "blurred"]),
    ("How can high blood pressure be prevented or controlled?",
     "Eat a healthy diet low in salt, keep a healthy weight, be physically active, limit alcohol, do not smoke, manage "
     "stress and take prescribed medicines.",
     ["salt|sodium", "weight", "physical|exercise", "alcohol"]),
    ("What are common symptoms of migraine?",
     "Migraine causes throbbing, pulsing headache often on one side, with nausea or vomiting and sensitivity to light "
     "and sound; some people have an aura.",
     ["nausea|vomit", "light", "aura|one side"]),
    ("How is the common cold treated?",
     "There is no cure for the common cold; rest, fluids and over-the-counter medicines help relieve symptoms, and "
     "antibiotics do not help.",
     ["rest", "fluid", "antibiotic|no cure"]),
    ("What is polycystic ovary syndrome?",
     "PCOS is a common hormonal condition in women of reproductive age causing irregular periods, excess androgen and "
     "polycystic ovaries; it is linked to infertility, insulin resistance and type 2 diabetes.",
     ["hormon", "irregular|period", "insulin|diabetes"]),
    ("What are the main risk factors for heart disease?",
     "Unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol, which raise blood pressure, blood "
     "glucose, blood lipids and weight.",
     ["diet", "physical|inactivity", "tobacco|smok", "alcohol"]),
]

# (query, expected patient name) - patient-memory retrieval (FAISS) hit@k
PATIENT_RETRIEVAL_CASES = [
    ("patient with sharp chest pain at the sternum after starting workouts", "Rebeca Nagle"),
    ("dry cough and mild fever treated with antihistamines", "Anjali Mehra"),
    ("metformin dose increased, increased thirst and urination", "David Thompson"),
    ("telmisartan for essential hypertension routine checkup", "Ramesh Kulkarni"),
    ("eGFR falling, albuminuria, started empagliflozin", "Suresh Negi"),
    ("pap smear and flu shot at well woman exam", "Rebeca Nagle"),
]

# (query, substring expected in a retrieved source URL) - medical knowledge retrieval hit@k
KNOWLEDGE_RETRIEVAL_CASES = [
    ("chronic kidney disease treatment options", "kidney"),
    ("how to lower high blood pressure", "blood"),
    ("migraine triggers and symptoms", "migraine|headache"),
    ("PCOS symptoms", "polycystic"),
    ("asthma inhaler treatment", "asthma"),
    ("flu vaccine and symptoms", "flu|influenza"),
]

# Booking scenarios: (patient name, specialty, preferred_date, time_of_day, booking expected to succeed)
# Date hints are week-level so outcomes do not depend on the weekday the evaluation runs.
BOOKING_CASES = [
    ("Ramesh Kulkarni", "Cardiology", None, None, True),
    ("David Thompson", "Endocrinology", "next week", "morning", True),
    ("Anjali Mehra", "Pulmonology", None, "morning", True),
    ("Rebeca Nagle", "Rheumatology", "next week", None, True),
    ("Suresh Negi", "Nutrition & Dietetics", "next week", None, True),
    ("Rahul Negi", "Dermatology", "next week", "afternoon", True),
    ("Rebeca Nagle", "Neurology", None, "afternoon", True),
    ("David Thompson", "Pulmonology", "next week", "evening", False),   # only 10:00-14:00 clinics -> graceful failure
]
