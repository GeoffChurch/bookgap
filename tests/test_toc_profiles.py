from bookgap.toc.profiles import _clean_title
from bookgap.toc.profiles import chapter_n_profile, PROFILES
from bookgap.toc.profiles import arabic_list_profile
from bookgap.toc.profiles import roman_list_profile


def _leaf_titles(parsed):
    def _walk(nodes):
        out = []
        for n in nodes:
            out += _walk(n.children) if n.children else [n.title]
        return out
    return _walk(parsed.nodes)


def _part_labels(parsed):
    return [n.title for n in parsed.nodes if n.kind == "part"]


def test_clean_title_strips_trailing_page_number():
    assert _clean_title("Introduction 4") == "Introduction"


def test_clean_title_strips_html_and_missing_and_nbsp():
    assert _clean_title("Special Relativity<br/>") == "Special Relativity"
    assert _clean_title("World Building [[MISSING]]") == "World Building"
    assert _clean_title("Coming clean:\xa0Framing") == "Coming clean: Framing"


def test_clean_title_strips_trailing_part_marker():
    assert _clean_title(
        "Checking Out Visitors: Meteors, Comets, and Man-Made Moons PART II: On"
    ) == "Checking Out Visitors: Meteors, Comets, and Man-Made Moons"


_PEARSON = ("INTRODUCTION: Writing in English PART I: PARAGRAPH ORGANIZATION "
            "CHAPTER 1 The Topic Sentence CHAPTER 2 Supporting Sentences "
            "CHAPTER 3 The Concluding Sentence PART II: BASIC TYPES")

def test_chapter_n_extracts_ordered_chapters():
    # The text has two real PART markers, so this now nests (part-aware);
    # the 3 CHAPTER entries land under PART I, and PART II is a childless leaf.
    p = chapter_n_profile(_PEARSON)
    assert p is not None and p.profile == "chapter-n"
    assert _part_labels(p) == ["PARAGRAPH ORGANIZATION", "BASIC TYPES"]
    assert _leaf_titles(p) == [
        "The Topic Sentence", "Supporting Sentences", "The Concluding Sentence",
        "BASIC TYPES"]

def test_chapter_n_rejects_fewer_than_three():
    assert chapter_n_profile("CHAPTER 1 Alpha CHAPTER 2 Beta") is None

def test_chapter_n_rejects_out_of_order():
    assert chapter_n_profile("CHAPTER 1 A CHAPTER 3 B CHAPTER 2 C CHAPTER 4 D") is None

def test_chapter_n_registered():
    assert PROFILES["chapter-n"] is chapter_n_profile


_POWER = ("1. Power Electronics 2. Power Diodes 3. Power Transistors "
          "4. Thyristor Devices 5. Single-Phase Uncontrolled Rectifiers")

def test_arabic_list_extracts_ordered_prefix():
    p = arabic_list_profile(_POWER)
    assert p is not None and p.profile == "arabic-list"
    assert [n.title for n in p.nodes] == [
        "Power Electronics", "Power Diodes", "Power Transistors",
        "Thyristor Devices", "Single-Phase Uncontrolled Rectifiers"]

def test_arabic_list_rejects_prose_enumeration_below_five():
    # A three-item "contributions" enumeration must not parse.
    assert arabic_list_profile(
        "We make three contributions: 1. a dataset 2. a model 3. results") is None

def test_arabic_list_ignores_decimal_subsections():
    # "1.1"/"1.2" are decimals, not top-level items; no ordered 1..5 list here.
    assert arabic_list_profile("Section 1.1 intro 1.2 setup 1.3 method") is None

def test_arabic_list_rejects_sentence_length_titles():
    long = " ".join(
        f"{i}. " + ("word " * 30) for i in range(1, 6))  # 5 items but ~150-char titles
    assert arabic_list_profile(long) is None


_ROUTLEDGE = ("I. Newtonian Mechanics<br/> II. Special Relativity<br/> "
              "III. General Relativity<br/> IV. Non-Relativistic Quantum Theory<br/> "
              "V. Quantum Field Theory<br/>")

def test_roman_list_extracts_ordered_and_cleans_html():
    p = roman_list_profile(_ROUTLEDGE)
    assert p is not None and p.profile == "roman-list"
    assert [n.title for n in p.nodes] == [
        "Newtonian Mechanics", "Special Relativity", "General Relativity",
        "Non-Relativistic Quantum Theory", "Quantum Field Theory"]

def test_roman_list_rejects_below_five():
    assert roman_list_profile("I. A II. B III. C IV. D") is None


def test_chapter_n_handles_comma_before_label():
    # Real Task-8 false-split: no whitespace, a comma directly precedes the label.
    # Old boundary class excluded ',', so all three labels were missed -> None.
    ab = "Signal Processing,Chapter 1: Fundamentals,Chapter 2: Systems,Chapter 3: Advanced"
    p = chapter_n_profile(ab)
    assert p is not None
    assert [n.title for n in p.nodes] == ["Fundamentals", "Systems", "Advanced"]


def test_clean_title_strips_trailing_comma():
    assert _clean_title("Concept of the PC,") == "Concept of the PC"

def test_clean_title_strips_stuck_page_number():
    assert _clean_title("Evolve or Die11") == "Evolve or Die"      # glued to lowercase
    assert _clean_title("Systems Integration 27") == "Systems Integration"
    assert _clean_title("CO2") == "CO2"                            # uppercase-preceded digit KEPT

def test_clean_title_strips_trailing_chapter_header():
    assert _clean_title("Authority Rules Chapter") == "Authority Rules"
    assert _clean_title("Misconduct Chapter 3") == "Misconduct"
    assert _clean_title("A New Chapter in History") == "A New Chapter in History"  # mid-title KEPT

def test_chapter_n_rejects_prose_in_chapter_enumeration():
    # "In Chapter N, <lowercase prose>" -> fragments start lowercase -> rejected as prose.
    ab = ("In Chapter 1, a microbial fuel cell was studied. "
          "In Chapter 2, an autonomous model was built. "
          "In Chapter 3, studies of dehumidification are presented.")
    assert chapter_n_profile(ab) is None


# ---------------------------------------------------------------------------
# Part-aware exemplars (validated against 8 real OpenAlex book abstracts).
# ---------------------------------------------------------------------------

_CRYPTOGRAPHY = (
    "NOTATION PREFACE CHAPTER 0 READER'S GUIDE CHAPTER 1 OVERVIEW PART ONE SYMMETRIC CIPHERS "
    "CHAPTER 2 CLASSICAL ENCRYPTION TECHNIQUES CHAPTER 3 BLOCK CIPHERS AND THE DATA ENCRYPTION "
    "STANDARD CHAPTER 4 INTRODUCTION TO FINITE FIELDS CHAPTER 5 ADVANCED ENCRYPTION STANDARD "
    "CHAPTER 6 MORE ON SYMMETRIC CIPHERS CHAPTER 7 CONFIDENTIALITY USING SYMMETRIC ENCRYPTION "
    "PART TWO PUBLIC-KEY ENCRYPTION AND HASH FUNCTIONS CHAPTER 8 INTRODUCTION TO NUMBER THEORY "
    "CHAPTER 9 PUBLIC-KEY CRYPTOGRAPHY AND RSA CHAPTER 10 KEY MANAGEMENT OTHER PUBLIC-KEY "
    "CRYPTOSYSTEMS CHAPTER 11 MESSAGE AUTHENTICATION AND HASH FUNCTIONS 1 CHAPTER 12 HASH AND MAC "
    "ALGORITHMS CHAPTER 13 DIGITAL SIGNATURES AND AUTHENTICATION PROTOCOLS PART THREE NETWORK "
    "SECURITY PRACTICE CHAPTER 14 AUTHENTICATION APPLICATIONS CHAPTER 15 ELECTRONIC MAIL SECURITY "
    "CHAPTER 16 IP SECURITY CHAPTER 17 WEB SECURITY PART FOUR SYSTEM SECURITY CHAPTER 18 INTRUDERS "
    "CHAPTER 19 MALICIOUS SOFTWARE CHAPTER 20 FIREWALLS APPENDICES APPENDIX A STANDARDS AND "
    "STANDARD-SETTING ORGANIZATIONS APPENDIX B PROJECTS FOR TEACHING CRYPTOGRAPHY AND NETWORK "
    "SECURITY ONLINE APPENDICES APPENDIX C SIMPLIFIED DES APPENDIX D THE MEANING OF mod APPENDIX "
    "E MORE ON SIMPLIFIED AES APPENDIX F KNAPSACK PUBLIC-KEY ALGORITHM APPENDIX G PROOF OF THE "
    "DIGITAL SIGNATURE ALGORITHM GLOSSARY REFERENCES INDEX LIST OF ACRONYMS")


def test_cryptography_parts_and_chapters():
    p = chapter_n_profile(_CRYPTOGRAPHY)
    assert p is not None and p.profile == "chapter-n"
    leaves = _leaf_titles(p)
    assert len(leaves) == 21
    assert _part_labels(p) == [
        "SYMMETRIC CIPHERS", "PUBLIC-KEY ENCRYPTION AND HASH FUNCTIONS",
        "NETWORK SECURITY PRACTICE", "SYSTEM SECURITY"]
    assert leaves[:2] == ["READER'S GUIDE", "OVERVIEW"]  # pre-part leaves
    part1 = next(n for n in p.nodes if n.title == "SYMMETRIC CIPHERS")
    assert part1.children[0].title == "CLASSICAL ENCRYPTION TECHNIQUES"
    assert not any(t.upper().startswith("APPENDIX") for t in leaves)


_BIODYNAMICS = (
    "Preface.PART I: INTRODUCTION TO SCIENCE.Chapter 1. Ways of Knowing.Chapter 2. Ways of "
    "Thinking.Chapter 3. Mechanics-A First Look.PART II: DYNAMICS, THE FUNDAMENTALS OF MOTION."
    "Chapter 4. Measures of Motion.Chapter 5. Energy-The Source of It All.Chapter 6. Linear "
    "Physics.Chapter 7. Biomechanics 1-Rectilinear Motion.Chapter 8. Biomechanics 2-Rotational "
    "Motion.PART III: FIELDS-HOW COMPLEX SYSTEMS MOVE.Chapter 9. Bioelectricity-Signals in the "
    "Body.Chapter 10. Molecules and Metabolism-Unseen Causes.Chapter 11. Bioenergetics-The "
    "Dynamics of Heat.Chapter 12. Fluids at Rest.Chapter 13. Fluids in Motion.PART IV: DATA "
    "ANALYSIS-WHAT WE CAN LEARN FROM MEASUREMENTS.Chapter 14. Biostatistics 1-When Things "
    "Converge.Chapter 15. Biostatistics 2-When Things Diverge.Chapter 16. Biodynamic Time Series "
    "1-Stride Rate Variability.Chapter 17. Biodynamic Time Series 2-Heart Rate Variability."
    "Chapter 18. Biodynamic Time Series 3-Breathing Rate Variability.Epilogue.PART V: "
    "MICROBIOGRAPHIES-THE SCIENTISTS WHO MADE HISTORY.References.Index.")


def test_biodynamics_restart_tolerant_parts():
    p = chapter_n_profile(_BIODYNAMICS)
    assert p is not None and p.profile == "chapter-n"
    leaves = _leaf_titles(p)
    assert len(leaves) == 19
    labels = _part_labels(p)
    assert len(labels) == 5
    assert labels[0] == "INTRODUCTION TO SCIENCE"
    assert labels[-1] == "MICROBIOGRAPHIES-THE SCIENTISTS WHO MADE HISTORY"
    part5 = p.nodes[-1]
    assert part5.kind == "part" and part5.children == ()  # childless leaf
    part1 = p.nodes[0]
    assert part1.title == "INTRODUCTION TO SCIENCE"
    assert part1.children[0].title == "Ways of Knowing"


_DRIVING_ECONOMY = (
    "Preface -- Part I. E-Governance -- Part II. Innovation Management -- Part III. Technology "
    "and Emerging Markets -- Part IV. Technology and Supply Chain Management -- Part V. Human "
    "Resource Management -- Part VI. Learning and Knowledge Management -- Part VII. Technology "
    "Management -- Part VIII. Technology and Development -- Part IX. IT Offshoring -- Part X. "
    "Technology and Innovation -- Part XI. Technology and Sustainability -- Part XII. Technology "
    "and Market -- Part XIII. Technology Nurturing -- Part XIV. New Product and Services "
    "Development.")


def test_driving_economy_parts_only_childless_leaves():
    p = chapter_n_profile(_DRIVING_ECONOMY)  # no chapter/arabic/roman markers exist -> any wins
    assert p is not None
    leaves = _leaf_titles(p)
    assert len(leaves) == 14
    assert leaves == [
        "E-Governance", "Innovation Management", "Technology and Emerging Markets",
        "Technology and Supply Chain Management", "Human Resource Management",
        "Learning and Knowledge Management", "Technology Management",
        "Technology and Development", "IT Offshoring", "Technology and Innovation",
        "Technology and Sustainability", "Technology and Market", "Technology Nurturing",
        "New Product and Services Development"]
    assert all(n.kind == "part" and n.children == () for n in p.nodes)


_ART_OF_TEACHING = (
    "SECTION I. BUILDING A PLACE FOR READING. 1. Co-Authoring a Literate Community in the "
    "Classroom. 2. Words That Change the World: Developing a Classroom Library. SECTION II. "
    "ONGOING STRUCTURES IN THE READING CURRICULUM. 3. Reading Aloud. 4. The Independent Reading "
    "Workshop. 5. Minilessons. 6. Coaching and Conferring with Readers. 7. Teaching Readers "
    "Within a Leveled Classroom Library. 8. Bringing Reading Assessment into the Very Real World "
    "of Classroom Teaching. 9. Supporting Readers Who Struggle with Meaning, Print, or Fluency. "
    "10. Guided Reading and Strategy Lessons. 11. Phonics and Word Study in the Primary Grades. "
    "12. A Curriculum of Talk. SECTION III. UNITS OF STUDY IN A READING WORKSHOP. Introduction: "
    "Units of Study in a Reading Curriculum. 13. September in a K-1 Reading Workshop. 14. "
    "Bringing in the Print Work of Reading. 15. Talking and Thinking About Books: Comprehension "
    "for Young Readers. 16. Reading Centers in the K-1 Classroom. 17. September in a 2-8 Grade "
    "Reading Workshop: Reading with Stamina and Comprehension. 18. Talking and Writing to Develop "
    "Ideas. 19. Reading Projects. 20. Book Clubs. SECTION IV. TEACHING THE QUALITIES OF GOOD "
    "READING. 21. Nonfiction Reading: The Words of Our World. 22. The Elements of Story. 23. "
    "Giving Our Students Tools and Strategies to Interpret Texts. 24. Writing About Reading. 25. "
    "Personal Response. 26. We Are the Authors of Our Lives. Appendix A: Examples of Leveled "
    "Reading Books. Appendix B: Read-Aloud Book Recommendations. Appendix C: Magazines for Kids. "
    "Bibliography. Children's Bibliography. Credits. Index.")


def test_art_of_teaching_arabic_sections():
    p = arabic_list_profile(_ART_OF_TEACHING)
    assert p is not None and p.profile == "arabic-list"
    leaves = _leaf_titles(p)
    assert len(leaves) == 26
    labels = _part_labels(p)
    assert len(labels) == 4
    assert labels[2] == "UNITS OF STUDY IN A READING WORKSHOP"  # no "Introduction:" run-on


_SOILS = (
    "I. SOIL MATERIALS. 1. Physical Character of Soil Constituents. 2. Natural Soil Deposits. 3. "
    "Soil Index Properties. 4. Soil Classification. 5. Stress Analysis and Engineering "
    "Properties. II. EARTHWORK IN THE CONSTRUCTION CONTRACT. 6. The Contract and Contract "
    "Documents. 7. Interpretation of Soil Reports. 8. Embankment Construction and Control. 9. "
    "Dewatering. 10. Excavations and Excavation Supports. 11. Foundation Construction. 12. "
    "Construction Access and Haul Roads. III. LABORATORY TESTING EXERCISES. 1. Calibration of "
    "Sand Cone and Density Sand, and Determination of Field Unit Weight of Soils. 2. Preparation "
    "of Soil Samples and Performance of Particle-Size Analyses. 3. Specific Gravity of Soils. 4. "
    "Soil Consistency Tests. 5. Compaction Tests. 6. Relative Density. 7. Falling-Head "
    "Permeability Test in Sand. 8. Direct Shear Test on Sand. 9. Unconfined Compression Test for "
    "Cohesive Soils. 10. Triaxial Compression Test on Sand. 11. Consolidation Test. Glossary of "
    "Terms. Problems Solutions. Index.")


def test_soils_flat_per_part_restart_dropped():
    p = arabic_list_profile(_SOILS)
    assert p is not None and p.profile == "arabic-list"
    leaves = _leaf_titles(p)
    assert len(leaves) == 12
    assert _part_labels(p) == []  # bare roman numerals ("I.", "II.") are not keyword parts
    assert leaves[2] == "Soil Index Properties"  # NOT truncated to "Soil"
    assert "Calibration of Sand Cone and Density Sand, and Determination of Field Unit Weight of "\
        "Soils" not in leaves  # Part III's restarted exercise list is dropped


_MASTERING_DATA_MODELING = (
    "Foreword. Preface. 1. Introduction. 2. Good Habits. 3. Reading an LDS with Sentences. 4. "
    "Vocabulary of LDS. 5. Visualizing Allowed and Disallowed Instances. 6. A Conversation with "
    "Users about Creatures and Skills. 7. Introduction to Mastering Shapes. 8. One-Entity, "
    "No-Relationship Shapes. 9. One-Attribute Shapes. 10. Two-Entity Shapes. 11. Shapes with More "
    "Than Two Entities. 12. Shapes with Reflexive Relationships. 13. LDS Syntax Rules. 14. "
    "Getting the Names Right. 15. Official Name. 16. Labeling Links. 17. Documenting an LDS. 18. "
    "Script for Controlled Evolution. 19. Local, Anytime Steps of Controlled Evolution. 20. "
    "Global, Anytime Steps of Controlled Evolution. 21. Conversations about Dairy Farming. 22. "
    "Constraints. 23. LDS for LDS. 24: Decisions: Designing a Data-Modeling Notation. 25. LDS and "
    "the Relational Model. 26: Cookbook: Recipes for Data Modelers. Appendix: Exercises for "
    "Mastery. Index.")


def test_mastering_data_modeling_colon_delimiter():
    p = arabic_list_profile(_MASTERING_DATA_MODELING)
    assert p is not None and p.profile == "arabic-list"
    leaves = _leaf_titles(p)
    assert len(leaves) == 26
    assert _part_labels(p) == []
    assert leaves[-4:] == [
        "LDS for LDS", "Decisions: Designing a Data-Modeling Notation",
        "LDS and the Relational Model", "Cookbook: Recipes for Data Modelers"]


_CONSTRUCTION_MGMT = (
    "ch. 1. Demolition -- ch. 2. Shoring and underpinning -- ch. 3. Mass excavation and site "
    "grading -- ch. 4. Dewatering -- ch. 5. Below-grade waterproofing -- ch. 6. Reinforcing "
    "steel -- ch. 7. Formwork -- ch. 8. Concrete placing and finishing -- ch. 9. Shotcrete -- ch. "
    "10. Structural steel -- ch. 11. Metal decking -- ch. 12. Miscellaneous metals -- ch. 13. "
    "Expansion joint covers -- ch. 14. Spray-applied fireproofing -- ch. 15. Above-grade "
    "waterproofing -- ch. 16. Lath and plaster -- ch. 17. Precast concrete -- ch. 18. Masonry -- "
    "ch. 19. Metal panels -- ch. 20. Roofing -- ch. 21. Flashing and louvers -- ch. 22. Glass and "
    "glazing -- ch. 23. Caulking and sealants -- ch. 24. Framing and drywall -- ch. 25. Building "
    "insulation and fire safing -- ch. 26. Doors, frames, and hardware -- ch. 27. Coiling doors "
    "and grilles -- ch. 28. Casework and millwork -- ch. 29. Painting and wall coverings -- ch. "
    "30. Ceramic tile and stone flooring -- ch. 31. Terrazzo and epoxy flooring -- ch. 32. "
    "Acoustical treatment -- ch. 33. Carpeting and resilient flooring -- ch. 34. Wood flooring -- "
    "ch. 35. Miscellaneous specialty work -- ch. 36. Food service equipment -- ch. 37. "
    "Elevators -- ch. 38. Plumbing -- ch. 39. Fire sprinklers -- ch. 40. Mechanical (HVAC) -- ch. "
    "41. Electrical -- ch. 42. Site utilities -- ch. 43. Asphaltic-concrete paving -- ch. 44. "
    "Site concrete -- ch. 45. Chain link fencing -- ch. 46. Landscaping and irrigation -- ch. 47. "
    "Subcontractor scope issues -- ch. 48. General contractor responsibilities.")


def test_construction_mgmt_flat_clean_titles():
    p = chapter_n_profile(_CONSTRUCTION_MGMT)
    assert p is not None and p.profile == "chapter-n"
    leaves = _leaf_titles(p)
    assert len(leaves) == 48
    assert _part_labels(p) == []
    assert "Demolition" in leaves
    assert "Shoring and underpinning" in leaves
    assert not any("--" in t or t.lower().endswith(" ch") for t in leaves)


_ACCESS_2002 = (
    "Introduction. You Don't Need to Be a Nerd to Use This Book. Sneaking a Peek at What's to "
    "Come. Part I: Which Came First: The Data or the Base? Part II: Truly Tempting Tables. Part "
    "III: Finding the Ultimate Answer to Everything (Well, Not Everything). Part IV: Turning Your "
    "Table into a Book. Part V: Wizards, Forms, and Other Mystical Stuff. Part VI: The Part of "
    "Tens. What the Funny Text Means. Finding Points of Interest. Setting Sail on the Voyage. "
    "Part I: Which Came First: The Data or the Base? Chapter 1: The 37-Minute Overview. In the "
    "Beginning, There Was Access 2002 (But It Wasn't Running). Chapter 2: Finding Your Way around "
    "like a Native. Making Sense of the Sights. Chapter 3: Calling the Online Saint Bernard and "
    "Other Forms of Help. Pressing F1 for Assistance. Part II: Truly Tempting Tables. Chapter 4: "
    "Designing and Building a Home for Your Data. Database Terms to Know and Tolerate. Chapter 5: "
    "Relationships, Keys, and Indexes (And Why You Really Do Care). The Joy (And Necessity) of a "
    "Primary Key. Chapter 6: New Data, Old Data, and Data in Need of Repair. Dragging Your Table "
    "into the Digital Workshop. Chapter 7: Making Your Table Think with Formats, Masks, and "
    "Validations. Finding the Place to Make a Change. Chapter 8: Making Your Datasheets Dance. "
    "Wandering Here, There, and Everywhere. Chapter 9: Table Remodeling Tips for the "
    "Do-It-Yourselfer. This Chapter Can Be Hazardous to Your Table's Design. Part III: Finding the "
    "Ultimate Answer to Everything (Well, Not Everything). Chapter 10: Quick Searches: Find, "
    "Filter, and Sort. Finding Stuff in Your Tables. Chapter 11: Pose a Simple Query, Get 10,000 "
    "Answers. Database Interrogation for Fun and Profit. Chapter 12: Searching a Slew of Tables. "
    "Some General Thoughts about Multiple-Table Queries. Chapter 13: The Ands and Ors of Dr. "
    "Boole. Comparing AND to OR. Chapter 14: Teaching Queries to Think and Count. Totaling "
    "Everything in Sight. Chapter 15: Calculating Your Way to Fame and Fortune. A Simple "
    "Calculation. Chapter 16: Automated Editing for Big Changes. First, This Word from Our "
    "Paranoid Sponsor. Part IV: Turning Your Table into a Book. Chapter 17: AutoReport: Like the "
    "Model-T, It's Clunky but It Runs. AutoReport Basics for High-Speed Information. Chapter 18: "
    "Wizardly Help with Labels, Charts, and Multilevel Reports. Creating Labels. Chapter 19: It's "
    "Amazing What a Little Formatting Can Do. Taking Your Report to the Design View Tune-Up Shop. "
    "Chapter 20: Headers and Footers for Groups, Pages, and Even (Egad) Whole Reports. Everything "
    "in Its Place. Part V: Wizards, Forms, and Other Mystical Stuff Chapter 21: Spinning Your "
    "Data into (And onto) the Web. Access 2002 and the Internet: A Match Made in Redmond. Chapter "
    "22: Making Forms That Look Cool and Work Great. Tax Forms and Data Forms Are Very Different "
    "Animals. Chapter 23: If Love Is Universal, Why Can't I Export to It? Importing Only the Best "
    "Information for Your Databases. Chapter 24: The Analyzer: Your Data's Dr. Freud, Dr. Watson, "
    "and Dr. Jekyll. It Slices, It Dices, It Builds Relational Databases! Chapter 25: Talking To "
    "Your Computer. What Is Speech Recognition (And What Can I Do with It)? Part VI: The Part of "
    "Tens. Chapter 26: Ten Timesaving Keyboard Shortcuts. Chapter 27: Ten Common Crises and How "
    "to Survive Them. Index.")


def test_access_2002_deduped_parts():
    p = chapter_n_profile(_ACCESS_2002)
    assert p is not None and p.profile == "chapter-n"
    leaves = _leaf_titles(p)
    assert len(leaves) == 27
    labels = _part_labels(p)
    assert len(labels) == 6  # deduped -- the abstract repeats its own ToC verbatim
    assert len(labels) == len(set(l.lower() for l in labels))  # no duplicate part labels
