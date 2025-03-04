import uuid
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

from codegate.db import connection
from codegate.muxing.semantic_router import PersonaDoesNotExistError, SemanticRouter


@pytest.fixture
def db_path():
    """Creates a temporary database file path."""
    current_test_dir = Path(__file__).parent
    db_filepath = current_test_dir / f"codegate_test_{uuid.uuid4()}.db"
    db_fullpath = db_filepath.absolute()
    connection.init_db_sync(str(db_fullpath))
    yield db_fullpath
    if db_fullpath.is_file():
        db_fullpath.unlink()


@pytest.fixture()
def db_recorder(db_path) -> connection.DbRecorder:
    """Creates a DbRecorder instance with test database."""
    return connection.DbRecorder(sqlite_path=db_path, _no_singleton=True)


@pytest.fixture()
def db_reader(db_path) -> connection.DbReader:
    """Creates a DbReader instance with test database."""
    return connection.DbReader(sqlite_path=db_path, _no_singleton=True)


@pytest.fixture()
def semantic_router_mocked_db(
    db_recorder: connection.DbRecorder, db_reader: connection.DbReader
) -> SemanticRouter:
    """Creates a SemanticRouter instance with mocked database."""
    semantic_router = SemanticRouter()
    semantic_router._db_reader = db_reader
    semantic_router._db_recorder = db_recorder
    return semantic_router


@pytest.mark.asyncio
async def test_add_persona(semantic_router_mocked_db: SemanticRouter):
    """Test adding a persona to the database."""
    persona_name = "test_persona"
    persona_desc = "test_persona_desc"
    await semantic_router_mocked_db.add_persona(persona_name, persona_desc)
    retrieved_persona = await semantic_router_mocked_db._db_reader.get_persona_by_name(persona_name)
    assert retrieved_persona.name == persona_name
    assert retrieved_persona.description == persona_desc


@pytest.mark.asyncio
async def test_persona_not_exist_match(semantic_router_mocked_db: SemanticRouter):
    """Test checking persona match when persona does not exist"""
    persona_name = "test_persona"
    query = "test_query"
    with pytest.raises(PersonaDoesNotExistError):
        await semantic_router_mocked_db.check_persona_match(persona_name, query)


class PersonaMatchTest(BaseModel):
    persona_name: str
    persona_desc: str
    pass_queries: List[str]
    fail_queries: List[str]


simple_persona = PersonaMatchTest(
    persona_name="test_persona",
    persona_desc="test_desc",
    pass_queries=["test_desc", "test_desc2"],
    fail_queries=["foo"],
)

software_architect = PersonaMatchTest(
    persona_name="software architect",
    persona_desc="""
        Expert in designing large-scale software systems and technical infrastructure.
        Specializes in distributed systems, microservices architecture,
        and cloud-native applications.
        Deep knowledge of architectural patterns like CQRS, event sourcing, hexagonal architecture,
        and domain-driven design.
        Experienced in designing scalable, resilient, and maintainable software solutions.
        Proficient in evaluating technology stacks and making strategic technical decisions.
        Skilled at creating architecture diagrams, technical specifications,
        and system documentation.
        Focuses on non-functional requirements like performance, security, and reliability.
        Guides development teams on best practices for implementing complex systems.
    """,
    pass_queries=[
        """
        How should I design a microservices architecture that can handle high traffic loads?
        """,
        """
        What's the best approach for implementing event sourcing in a distributed system?
        """,
        """
        I need to design a system that can scale to millions of users. What architecture would you
        recommend?
        """,
        """
        Can you explain the trade-offs between monolithic and microservices architectures for our
        new project?
        """,
    ],
    fail_queries=[
        """
        How do I create a simple landing page with HTML and CSS?
        """,
        """
        What's the best way to optimize my SQL query performance?
        """,
        """
        Can you help me debug this JavaScript function that's throwing an error?
        """,
        """
        How do I implement user authentication in my React application?
        """,
    ],
)

# Data Scientist Persona
data_scientist = PersonaMatchTest(
    persona_name="data scientist",
    persona_desc="""
        Expert in analyzing and interpreting complex data to solve business problems.
        Specializes in statistical analysis, machine learning algorithms, and predictive modeling.
        Builds and deploys models for classification, regression, clustering, and anomaly detection.
        Proficient in data preprocessing, feature engineering, and model evaluation techniques.
        Uses Python with libraries like NumPy, Pandas, scikit-learn, TensorFlow, and PyTorch.
        Experienced with data visualization using Matplotlib, Seaborn, and interactive dashboards.
        Applies experimental design principles and A/B testing methodologies.
        Works with structured and unstructured data, including time series and text.
        Implements data pipelines for model training, validation, and deployment.
        Communicates insights and recommendations based on data analysis to stakeholders.

        Handles class imbalance problems in classification tasks using techniques like SMOTE,
        undersampling, oversampling, and class weighting. Addresses customer churn prediction
        challenges by identifying key features that indicate potential churners.

        Applies feature selection methods for high-dimensional datasets, including filter methods
        (correlation, chi-square), wrapper methods (recursive feature elimination), and embedded
        methods (LASSO regularization).

        Prevents overfitting and high variance in tree-based models like random forests through
        techniques such as pruning, setting maximum depth, adjusting minimum samples per leaf,
        and cross-validation.

        Specializes in time series forecasting for sales and demand prediction, using methods like
        ARIMA, SARIMA, Prophet, and exponential smoothing to handle seasonal patterns and trends.
        Implements forecasting models that account for quarterly business cycles and seasonal
        variations in customer behavior.

        Evaluates model performance using appropriate metrics: accuracy, precision, recall,
        F1-score
        for classification; RMSE, MAE, R-squared for regression; and specialized metrics for
        time series forecasting like MAPE and SMAPE.

        Experienced in developing customer segmentation models, recommendation systems,
        anomaly detection algorithms, and predictive maintenance solutions.
    """,
    pass_queries=[
        """
        How should I handle class imbalance in my customer churn prediction model?
        """,
        """
        What feature selection techniques would work best for my high-dimensional dataset?
        """,
        """
        I'm getting high variance in my random forest model. How can I prevent overfitting?
        """,
        """
        What's the best approach for forecasting seasonal time series data for our sales
        predictions?
        """,
    ],
    fail_queries=[
        """
        How do I structure my React components for a single-page application?
        """,
        """
        What's the best way to implement a CI/CD pipeline for my microservices?
        """,
        """
        Can you help me design a responsive layout for mobile and desktop browsers?
        """,
        """
        How should I configure my Kubernetes cluster for high availability?
        """,
    ],
)

# UX Designer Persona
ux_designer = PersonaMatchTest(
    persona_name="ux designer",
    persona_desc="""
        Expert in creating intuitive, user-centered digital experiences and interfaces.
        Specializes in user research, usability testing, and interaction design.
        Creates wireframes, prototypes, and user flows to visualize design solutions.
        Conducts user interviews, usability studies, and analyzes user feedback.
        Develops user personas and journey maps to understand user needs and pain points.
        Designs information architecture and navigation systems for complex applications.
        Applies design thinking methodology to solve user experience problems.
        Knowledgeable about accessibility standards and inclusive design principles.
        Collaborates with product managers and developers to implement user-friendly features.
        Uses tools like Figma, Sketch, and Adobe XD to create high-fidelity mockups.
    """,
    pass_queries=[
        """
        How can I improve the user onboarding experience for my mobile application?
        """,
        """
        What usability testing methods would you recommend for evaluating our new interface design?
        """,
        """
        I'm designing a complex dashboard. What information architecture would make it most
        intuitive for users?
        """,
        """
        How should I structure user research to identify pain points in our current
        checkout process?
        """,
    ],
    fail_queries=[
        """
        How do I configure a load balancer for my web servers?
        """,
        """
        What's the best way to implement a caching layer in my application?
        """,
        """
        Can you explain how to set up a CI/CD pipeline with GitHub Actions?
        """,
        """
        How do I optimize my database queries for better performance?
        """,
    ],
)

# DevOps Engineer Persona
devops_engineer = PersonaMatchTest(
    persona_name="devops engineer",
    persona_desc="""
        Expertise: Infrastructure automation, CI/CD pipelines, cloud services, containerization,
        and monitoring.
        Proficient with tools like Docker, Kubernetes, Terraform, Ansible, and Jenkins.
        Experienced with cloud platforms including AWS, Azure, and Google Cloud.
        Strong knowledge of Linux/Unix systems administration and shell scripting.
        Skilled in implementing microservices architectures and service mesh technologies.
        Focus on reliability, scalability, security, and operational efficiency.
        Practices infrastructure as code, GitOps, and site reliability engineering principles.
        Experienced with monitoring tools like Prometheus, Grafana, and ELK stack.
    """,
    pass_queries=[
        """
        What's the best way to set up auto-scaling for my Kubernetes cluster on AWS?
        """,
        """
        I need to implement a zero-downtime deployment strategy for my microservices.
        What approaches would you recommend?
        """,
        """
        How can I improve the security of my CI/CD pipeline and prevent supply chain attacks?
        """,
        """
        What monitoring metrics should I track to ensure the reliability of my distributed system?
        """,
    ],
    fail_queries=[
        """
        How do I design an effective user onboarding flow for my mobile app?
        """,
        """
        What's the best algorithm for sentiment analysis on customer reviews?
        """,
        """
        Can you help me with color theory for my website redesign?
        """,
        """
        I need advice on optimizing my SQL queries for a reporting dashboard.
        """,
    ],
)

# Security Specialist Persona
security_specialist = PersonaMatchTest(
    persona_name="security specialist",
    persona_desc="""
        Expert in cybersecurity, application security, and secure system design.
        Specializes in identifying and mitigating security vulnerabilities and threats.
        Performs security assessments, penetration testing, and code security reviews.
        Implements security controls like authentication, authorization, and encryption.
        Knowledgeable about common attack vectors such as injection attacks, XSS, CSRF, and SSRF.
        Experienced with security frameworks and standards like OWASP Top 10, NIST, and ISO 27001.
        Designs secure architectures and implements defense-in-depth strategies.
        Conducts security incident response and forensic analysis.
        Implements security monitoring, logging, and alerting systems.
        Stays current with emerging security threats and mitigation techniques.
    """,
    pass_queries=[
        """
        How can I protect my web application from SQL injection attacks?
        """,
        """
        What security controls should I implement for storing sensitive user data?
        """,
        """
        How do I conduct a thorough security assessment of our cloud infrastructure?
        """,
        """
        What's the best approach for implementing secure authentication in my API?
        """,
    ],
    fail_queries=[
        """
        How do I optimize the loading speed of my website?
        """,
        """
        What's the best way to implement responsive design for mobile devices?
        """,
        """
        Can you help me design a database schema for my e-commerce application?
        """,
        """
        How should I structure my React components for better code organization?
        """,
    ],
)

# Mobile Developer Persona
mobile_developer = PersonaMatchTest(
    persona_name="mobile developer",
    persona_desc="""
        Expert in building native and cross-platform mobile applications for iOS and Android.
        Specializes in mobile UI development, responsive layouts, and platform-specific
        design patterns.
        Proficient in Swift and SwiftUI for iOS, Kotlin for Android, and React Native or
        Flutter for cross-platform.
        Implements mobile-specific features like push notifications, offline storage, and
        location services.
        Optimizes mobile applications for performance, battery efficiency, and limited
        network connectivity.
        Experienced with mobile app architecture patterns like MVVM, MVC, and Redux.
        Integrates with device hardware features including camera, biometrics, sensors,
        and Bluetooth.
        Familiar with app store submission processes, app signing, and distribution workflows.
        Implements secure data storage, authentication, and API communication on mobile devices.
        Designs and develops responsive interfaces that work across different screen sizes
        and orientations.

        Implements sophisticated offline-first data synchronization strategies
        for mobile applications,
        handling conflict resolution, data merging, and background syncing when connectivity
        is restored.
        Uses technologies like Realm, SQLite, Core Data, and Room Database to enable seamless
        offline
        experiences in React Native and native apps.

        Structures Swift code following the MVVM (Model-View-ViewModel) architectural pattern
        to create
        maintainable, testable iOS applications. Implements proper separation of concerns
        with bindings
        between views and view models using Combine, RxSwift, or SwiftUI's native state management.

        Specializes in deep linking implementation for both Android and iOS, enabling app-to-app
        communication, marketing campaign tracking, and seamless user experiences when navigating
        between web and mobile contexts. Configures Universal Links, App Links, and custom URL
        schemes.

        Optimizes battery usage for location-based features by implementing intelligent location
        tracking
        strategies, including geofencing, significant location changes, deferred location updates,
        and
        region monitoring. Balances accuracy requirements with power consumption constraints.

        Develops efficient state management solutions for complex mobile applications using Redux,
        MobX, Provider, or Riverpod for React Native apps, and native state management approaches
        for iOS and Android.

        Creates responsive mobile interfaces that adapt to different device orientations,
        screen sizes,
        and pixel densities using constraint layouts, auto layout, size classes, and flexible
        grid systems.
    """,
    pass_queries=[
        """
        What's the best approach for implementing offline-first data synchronization in my mobile
        app?
        """,
        """
        How should I structure my Swift code to implement the MVVM pattern effectively?
        """,
        """
        What's the most efficient way to handle deep linking and app-to-app communication on
        Android?
        """,
        """
        How can I optimize battery usage when implementing background location tracking?
        """,
    ],
    fail_queries=[
        """
        How do I design a database schema with proper normalization for my web application?
        """,
        """
        What's the best approach for implementing a distributed caching layer in my microservices?
        """,
        """
        Can you help me set up a data pipeline for processing large datasets with Apache Spark?
        """,
        """
        How should I configure my load balancer to distribute traffic across my web servers?
        """,
    ],
)

# Database Administrator Persona
database_administrator = PersonaMatchTest(
    persona_name="database administrator",
    persona_desc="""
        Expert in designing, implementing, and managing database systems for optimal performance and
        reliability.
        Specializes in database architecture, schema design, and query optimization techniques.
        Proficient with relational databases like PostgreSQL, MySQL, Oracle, and SQL Server.
        Implements and manages database security, access controls, and data protection measures.
        Designs high-availability solutions using replication, clustering, and failover mechanisms.
        Develops and executes backup strategies, disaster recovery plans, and data retention
        policies.
        Monitors database performance, identifies bottlenecks, and implements optimization
        solutions.
        Creates and maintains indexes, partitioning schemes, and other performance-enhancing
        structures.
        Experienced with database migration, version control, and change management processes.
        Implements data integrity constraints, stored procedures, triggers, and database automation.

        Optimizes complex JOIN query performance in PostgreSQL through advanced techniques including
        query rewriting, proper indexing strategies, materialized views, and query plan analysis.
        Uses EXPLAIN ANALYZE to identify bottlenecks in query execution plans and implements
        appropriate optimizations for specific query patterns.

        Designs and implements high-availability MySQL configurations with automatic failover using
        technologies like MySQL Group Replication, Galera Cluster, Percona XtraDB Cluster, or MySQL
        InnoDB Cluster with MySQL Router. Configures synchronous and asynchronous replication
        strategies
        to balance consistency and performance requirements.

        Develops sophisticated indexing strategies for tables with frequent write operations and
        complex
        read queries, balancing write performance with read optimization. Implements partial
        indexes,
        covering indexes, and composite indexes based on query patterns and cardinality analysis.

        Specializes in large-scale database migrations between different database engines,
        particularly
        Oracle to PostgreSQL transitions. Uses tools like ora2pg, AWS DMS, and custom ETL processes
        to
        ensure data integrity, schema compatibility, and minimal downtime during migration.

        Implements table partitioning schemes based on data access patterns, including range
        partitioning
        for time-series data, list partitioning for categorical data, and hash partitioning for
        evenly
        distributed workloads.

        Configures and manages database connection pooling, query caching, and buffer management to
        optimize resource utilization and throughput under varying workloads.

        Designs and implements database sharding strategies for horizontal scaling, including
        consistent hashing algorithms, shard key selection, and cross-shard query optimization.
    """,
    pass_queries=[
        """
        How can I optimize the performance of complex JOIN queries in my PostgreSQL database?
        """,
        """
        What's the best approach for implementing a high-availability MySQL setup with automatic
        failover?
        """,
        """
        How should I design my indexing strategy for a table with frequent writes and complex read
        queries?
        """,
        """
        What's the most efficient way to migrate a large Oracle database to PostgreSQL with minimal
        downtime?
        """,
    ],
    fail_queries=[
        """
        How do I structure my React components to implement the Redux state management pattern?
        """,
        """
        What's the best approach for implementing responsive design with CSS Grid and Flexbox?
        """,
        """
        Can you help me set up a CI/CD pipeline for my containerized microservices?
        """,
    ],
)

# Natural Language Processing Specialist Persona
nlp_specialist = PersonaMatchTest(
    persona_name="nlp specialist",
    persona_desc="""
        Expertise: Natural language processing, computational linguistics, and text analytics.
        Proficient with NLP libraries and frameworks like NLTK, spaCy, Hugging Face Transformers,
        and Gensim.
        Experience with language models such as BERT, GPT, T5, and their applications.
        Skilled in text preprocessing, tokenization, lemmatization, and feature extraction
        techniques.
        Knowledge of sentiment analysis, named entity recognition, topic modeling, and text
        classification.
        Familiar with word embeddings, contextual embeddings, and language representation methods.
        Understanding of machine translation, question answering, and text summarization systems.
        Background in information retrieval, semantic search, and conversational AI development.
    """,
    pass_queries=[
        """
        What approach should I take to fine-tune BERT for my custom text classification task?
        """,
        """
        How can I improve the accuracy of my named entity recognition system for medical texts?
        """,
        """
        What's the best way to implement semantic search using embeddings from language models?
        """,
        """
        I need to build a sentiment analysis system that can handle sarcasm and idioms.
        Any suggestions?
        """,
    ],
    fail_queries=[
        """
        How do I optimize my React components to reduce rendering time?
        """,
        """
        What's the best approach for implementing a CI/CD pipeline with Jenkins?
        """,
        """
        Can you help me design a responsive UI for my web application?
        """,
        """
        How should I structure my microservices architecture for scalability?
        """,
    ],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona_match_test",
    [
        simple_persona,
        software_architect,
        data_scientist,
        ux_designer,
        devops_engineer,
        security_specialist,
        mobile_developer,
        database_administrator,
        nlp_specialist,
    ],
)
async def test_check_persona_match(
    semantic_router_mocked_db: SemanticRouter, persona_match_test: PersonaMatchTest
):
    """Test checking persona match."""
    await semantic_router_mocked_db.add_persona(
        persona_match_test.persona_name, persona_match_test.persona_desc
    )

    # Check for the queries that should pass
    for query in persona_match_test.pass_queries:
        match = await semantic_router_mocked_db.check_persona_match(
            persona_match_test.persona_name, query
        )
        assert match is True

    # Check for the queries that should fail
    for query in persona_match_test.fail_queries:
        match = await semantic_router_mocked_db.check_persona_match(
            persona_match_test.persona_name, query
        )
        assert match is False
