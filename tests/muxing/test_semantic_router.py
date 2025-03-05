import uuid
from pathlib import Path
from typing import List

import pytest
from pydantic import BaseModel

from codegate.db import connection
from codegate.muxing.semantic_router import (
    PersonaDoesNotExistError,
    PersonaSimilarDescriptionError,
    SemanticRouter,
)


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
async def test_add_duplicate_persona(semantic_router_mocked_db: SemanticRouter):
    """Test adding a persona to the database."""
    persona_name = "test_persona"
    persona_desc = "test_persona_desc"
    await semantic_router_mocked_db.add_persona(persona_name, persona_desc)

    # Update the description to not trigger the similarity check
    updated_description = "foo and bar description"
    with pytest.raises(connection.AlreadyExistsError):
        await semantic_router_mocked_db.add_persona(persona_name, updated_description)


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

# Architect Persona
architect = PersonaMatchTest(
    persona_name="architect",
    persona_desc="""
        Expert in designing and planning software systems, technical infrastructure, and solution
        architecture.
        Specializes in creating scalable, maintainable, and resilient system designs.
        Deep knowledge of architectural patterns, principles, and best practices.
        Experienced in evaluating technology stacks and making strategic technical decisions.
        Skilled at creating architecture diagrams, technical specifications, and system
        documentation.
        Focuses on both functional and non-functional requirements like performance, security,
        and reliability.
        Guides development teams on implementing complex systems and following architectural
        guidelines.

        Designs system architectures that balance business needs with technical constraints.
        Creates technical roadmaps and migration strategies for legacy system modernization.
        Evaluates trade-offs between different architectural approaches (monolithic, microservices,
        serverless).
        Implements domain-driven design principles to align software with business domains.

        Develops reference architectures and technical standards for organization-wide adoption.
        Conducts architecture reviews and provides recommendations for improvement.
        Collaborates with stakeholders to translate business requirements into technical solutions.
        Stays current with emerging technologies and evaluates their potential application.

        Designs for cloud-native environments using containerization, orchestration, and managed
        services.
        Implements event-driven architectures using message queues, event buses, and streaming
        platforms.
        Creates data architectures that address storage, processing, and analytics requirements.
        Develops integration strategies for connecting disparate systems and services.
    """,
    pass_queries=[
        """
        How should I design a system architecture that can scale with our growing user base?
        """,
        """
        What's the best approach for migrating our monolithic application to microservices?
        """,
        """
        I need to create a technical roadmap for modernizing our legacy systems. Where should
        I start?
        """,
        """
        Can you help me evaluate different cloud providers for our new infrastructure?
        """,
        """
        What architectural patterns would you recommend for a distributed e-commerce platform?
        """,
    ],
    fail_queries=[
        """
        How do I fix this specific bug in my JavaScript code?
        """,
        """
        What's the syntax for a complex SQL query joining multiple tables?
        """,
        """
        How do I implement authentication in my React application?
        """,
        """
        What's the best way to optimize the performance of this specific function?
        """,
    ],
)

# Coder Persona
coder = PersonaMatchTest(
    persona_name="coder",
    persona_desc="""
        Expert in full stack development, programming, and software implementation.
        Specializes in writing, debugging, and optimizing code across the entire technology stack.

        Proficient in multiple programming languages including JavaScript, Python, Java, C#, and
        TypeScript.
        Implements efficient algorithms and data structures to solve complex programming challenges.
        Develops maintainable code with appropriate patterns and practices for different contexts.

        Experienced in frontend development using modern frameworks and libraries.
        Creates responsive, accessible user interfaces with HTML, CSS, and JavaScript frameworks.
        Implements state management, component architecture,
        and client-side performance optimization for frontend applications.

        Skilled in backend development and server-side programming.
        Builds RESTful APIs, GraphQL services, and microservices architectures.
        Implements authentication, authorization, and security best practices in web applications.
        Understands best ways for different backend problems, like file uploads, caching,
        and database interactions.

        Designs and manages databases including schema design, query optimization,
        and data modeling.
        Works with both SQL and NoSQL databases to implement efficient data storage solutions.
        Creates data access layers and ORM implementations for application data requirements.

        Handles integration between different systems and third-party services.
        Implements webhooks, API clients, and service communication patterns.
        Develops data transformation and processing pipelines for various application needs.

        Identifies and resolves performance issues across the application stack.
        Uses debugging tools, profilers, and testing frameworks to ensure code quality.
        Implements comprehensive testing strategies including unit, integration,
        and end-to-end tests.
    """,
    pass_queries=[
        """
        How do I implement authentication in my web application?
        """,
        """
        What's the best way to structure a RESTful API for my project?
        """,
        """
        I need help optimizing my database queries for better performance.
        """,
        """
        How should I implement state management in my frontend application?
        """,
        """
        What's the differnce between SQL and NoSQL databases, and when should I use each?
        """,
    ],
    fail_queries=[
        """
        What's the best approach for setting up a CI/CD pipeline for our team?
        """,
        """
        Can you help me configure auto-scaling for our Kubernetes cluster?
        """,
        """
        How should I structure our cloud infrastructure for better cost efficiency?
        """,
        """
        How do I cook a delicious lasagna for dinner?
        """,
    ],
)

# DevOps/SRE Engineer Persona
devops_sre = PersonaMatchTest(
    persona_name="devops/sre engineer",
    persona_desc="""
        Expert in infrastructure automation, deployment pipelines, and operational reliability.
        Specializes in building and maintaining scalable, resilient, and secure infrastructure.
        Proficient with cloud platforms (AWS, Azure, GCP), containerization, and orchestration.
        Experienced with infrastructure as code, configuration management, and automation tools.
        Skilled in implementing CI/CD pipelines, monitoring systems, and observability solutions.
        Focuses on reliability, performance, security, and operational efficiency.
        Practices site reliability engineering principles and DevOps methodologies.

        Designs and implements cloud infrastructure using services like compute, storage,
        networking, and databases.
        Creates infrastructure as code using tools like Terraform, CloudFormation, or Pulumi.
        Configures and manages container orchestration platforms like Kubernetes and ECS.
        Implements CI/CD pipelines using tools like Jenkins, GitHub Actions, GitLab CI, or CircleCI.

        Sets up comprehensive monitoring, alerting, and observability solutions.
        Implements logging aggregation, metrics collection, and distributed tracing.
        Creates dashboards and visualizations for system performance and health.
        Designs and implements disaster recovery and backup strategies.

        Automates routine operational tasks and infrastructure maintenance.
        Conducts capacity planning, performance tuning, and cost optimization.
        Implements security best practices, compliance controls, and access management.
        Performs incident response, troubleshooting, and post-mortem analysis.

        Designs for high availability, fault tolerance, and graceful degradation.
        Implements auto-scaling, load balancing, and traffic management solutions.
        Creates runbooks, documentation, and operational procedures.
        Conducts chaos engineering experiments to improve system resilience.
    """,
    pass_queries=[
        """
        How do I set up a Kubernetes cluster with proper high availability?
        """,
        """
        What's the best approach for implementing a CI/CD pipeline for our microservices?
        """,
        """
        How can I automate our infrastructure provisioning using Terraform?
        """,
        """
        What monitoring metrics should I track to ensure the reliability of our system?
        """,
    ],
    fail_queries=[
        """
        How do I implement a sorting algorithm in Python?
        """,
        """
        What's the best way to structure my React components for a single-page application?
        """,
        """
        Can you help me design a database schema for my e-commerce application?
        """,
        """
        How do I create a responsive layout using CSS Grid and Flexbox?
        """,
        """
        What's the most efficient algorithm for finding the shortest path in a graph?
        """,
    ],
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona_match_test",
    [
        simple_persona,
        architect,
        coder,
        devops_sre,
    ],
)
async def test_check_persona_pass_match(
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona_match_test",
    [
        simple_persona,
        architect,
        coder,
        devops_sre,
    ],
)
async def test_check_persona_fail_match(
    semantic_router_mocked_db: SemanticRouter, persona_match_test: PersonaMatchTest
):
    """Test checking persona match."""
    await semantic_router_mocked_db.add_persona(
        persona_match_test.persona_name, persona_match_test.persona_desc
    )

    # Check for the queries that should fail
    for query in persona_match_test.fail_queries:
        match = await semantic_router_mocked_db.check_persona_match(
            persona_match_test.persona_name, query
        )
        assert match is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "personas",
    [
        [
            coder,
            devops_sre,
            architect,
        ]
    ],
)
async def test_persona_diff_description(
    semantic_router_mocked_db: SemanticRouter,
    personas: List[PersonaMatchTest],
):
    # First, add all existing personas
    for persona in personas:
        await semantic_router_mocked_db.add_persona(persona.persona_name, persona.persona_desc)

    last_added_persona = personas[-1]
    with pytest.raises(PersonaSimilarDescriptionError):
        await semantic_router_mocked_db.add_persona(
            "repeated persona", last_added_persona.persona_desc
        )
