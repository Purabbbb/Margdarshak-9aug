import AboutCard from './AboutCard.jsx'
import styles from '../pages/LandingPage.module.css'

export default function AboutGrid() {
  const cards = [
    {
      step: '01',
      title: 'Preprocessing & Data Ingestion',
      description: 'Extracts raw PDF text with pdfminer.six, then tokenizes, removes stopwords, and lemmatizes for clean downstream analysis.'
    },
    {
      step: '02',
      title: 'Hybrid Entity & Skill Parsing',
      description: 'Combines spaCy NER and regex for personal entities, plus exact and fuzzy matching to identify technical competencies.'
    },
    {
      step: '03',
      title: 'Semantic Role Classification',
      description: 'Ranks candidate fit by cosine similarity between extracted skills and O*NET-aligned role skill vectors.'
    },
    {
      step: '04',
      title: 'Quantified Gap Analysis',
      description: 'Uses set subtraction to compare current skills with target role requirements and identifies the top missing skills.'
    },
    {
      step: '05',
      title: 'Curated Resource Mapping',
      description: 'Maps each skill gap to free learning resources and certifications to build a personalized growth roadmap.'
    },
    {
      step: '06',
      title: 'Real-Time Market Matching',
      description: 'Queries Adzuna API to pull live job listings based on predicted role and location preferences.'
    },
    {
      step: '07',
      title: 'Intelligent Career Interaction',
      description: 'A lightweight ML intent-classification chatbot with contextual RAG supports instant, conversational career guidance.'
    }
  ]

  return (
    <div className={styles.aboutGrid}>
      {cards.map((card) => (
        <AboutCard
          key={card.step}
          step={card.step}
          title={card.title}
          description={card.description}
        />
      ))}

      <article className={styles.aboutVideoCard}>
        <video className={styles.aboutVideo} autoPlay muted loop playsInline>
          <source src="/Cinematic_Resume_Pile_Video_Generation.mp4" type="video/mp4" />
        </video>
      </article>
    </div>
  )
}