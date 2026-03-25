import re

with open('train_evaluate.py', encoding='utf-8') as f:
    content = f.read()

# Quitar train_ease y el import del final
content = re.sub(r'\nfrom src\.models\.ease_model import EASERecommender\n', '\n', content)
content = re.sub(r'\ndef train_ease\(a\):.*?return m\n', '\n', content, flags=re.DOTALL)

# Agregar el import al inicio junto a los otros imports
content = content.replace(
    'from src.models.als_model     import ALSRecommender',
    'from src.models.als_model     import ALSRecommender\nfrom src.models.ease_model    import EASERecommender'
)

# Insertar train_ease antes de train_content_based
ease_fn = (
    "\ndef train_ease(a):\n"
    "    logger.info('Entrenando: EASE')\n"
    "    m = EASERecommender(lambda_reg=500, top_n=TOP_N)\n"
    "    m.fit(build_train_matrix(a[\"events_clean\"])); m.save(); return m\n"
    "\n"
)
content = content.replace('def train_content_based', ease_fn + 'def train_content_based')

with open('train_evaluate.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("OK")
