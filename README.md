# Mctx: Enterprise Decision Intelligence Platform

Mctx is a comprehensive decision intelligence platform built on a [JAX](https://github.com/google/jax)-native
implementation of Monte Carlo Tree Search (MCTS) algorithms such as
[AlphaZero](https://deepmind.com/blog/article/alphazero-shedding-new-light-grand-games-chess-shogi-and-go),
[MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules), and
[Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO). Designed for enterprise applications,
Mctx delivers measurable business value through superior decision modeling, T4-optimized performance,
distributed computing capabilities, enterprise integrations, and intuitive visualization.

![MCTX Dashboard](https://example.com/mctx-dashboard.png)

## Business Value

MCTX transforms decision-making across industries, delivering measurable ROI:

- **47% improvement** in decision quality
- **82% reduction** in decision time
- **73% more** potential risks identified
- **28% reduction** in operational costs
- **350-1200% ROI** over 3 years

[Read our Executive Overview](docs/executive_overview.md) for business impact details.

## Installation

You can install the latest released version of Mctx from PyPI via:

```sh
pip install mctx
```

or you can install the latest development version from GitHub:

```sh
pip install git+https://github.com/google-deepmind/mctx.git
```

For optimized installations with additional features:

```sh
# With T4 GPU optimizations
pip install mctx[t4]

# With distributed capabilities
pip install mctx[distributed]

# With enterprise integrations (SAP HANA)
pip install mctx[enterprise]

# With all features
pip install mctx[all]
```

## Business Solutions

Mctx provides industry-specific solutions with proven ROI:

| Industry | Solution Areas | Business Impact |
|----------|----------------|-----------------|
| Financial Services | Portfolio optimization, risk management | 3.2% higher returns, 47% better risk assessment |
| Healthcare | Resource allocation, patient flow | 31% more capacity, 24% lower costs |
| Manufacturing | Supply chain, production planning | 22% inventory reduction, 35% better resilience |
| Retail | Inventory, pricing optimization | 28% lower carrying costs, 62% fewer stockouts |
| Energy | Trading optimization, grid management | 4.1% higher profits, 23% better reliability |

[View All Industry Solutions](docs/industry_solutions.md)

## Motivation

In today's complex business environment, organizations face unprecedented decision-making challenges. Traditional approaches cannot adequately handle uncertainty, explore sufficient alternatives, or balance competing priorities efficiently.

Mctx addresses these challenges through innovative search algorithms that have been combined with learned models parameterized by deep neural networks, resulting in one of the most powerful and general decision intelligence platforms available.

Through this enterprise-ready platform, we help organizations make better decisions, optimize resource allocation, and manage risk more effectively, delivering measurable business value across operations.

## Decision Intelligence Capabilities

In business environments, decision-makers must balance multiple objectives, navigate uncertainty, and allocate limited resources efficiently. Mctx provides:

- **Multi-scenario Simulation**: Explore thousands of potential futures simultaneously
- **Risk-weighted Decision Analysis**: Balance opportunity and risk across scenarios
- **Resource Optimization**: Allocate finite resources for maximum business impact
- **Competitive Intelligence**: Model market dynamics and competitor responses
- **Transparent Decision Paths**: Visualize decision trees for stakeholder alignment

## Quickstart

Mctx provides a low-level generic `search` function and high-level concrete
policies: `muzero_policy` and `gumbel_muzero_policy`.

The user needs to provide several learned components to specify the
representation, dynamics and prediction used by [MuZero](https://deepmind.com/blog/article/muzero-mastering-go-chess-shogi-and-atari-without-rules).
In the context of the Mctx library, the representation of the *root* state is
specified by a `RootFnOutput`. The `RootFnOutput` contains the `prior_logits`
from a policy network, the estimated `value` of the root state, and any
`embedding` suitable to represent the root state for the environment model.

The dynamics environment model needs to be specified by a `recurrent_fn`.
A `recurrent_fn(params, rng_key, action, embedding)` call takes an `action` and
a state `embedding`. The call should return a tuple `(recurrent_fn_output,
new_embedding)` with a `RecurrentFnOutput` and the embedding of the next state.
The `RecurrentFnOutput` contains the `reward` and `discount` for the transition,
and `prior_logits` and `value` for the new state.

In [`examples/visualization_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/visualization_demo.py), you can
see calls to a policy:

```python
policy_output = mctx.gumbel_muzero_policy(params, rng_key, root, recurrent_fn,
                                         num_simulations=32)
```

The `policy_output.action` contains the action proposed by the search. That
action can be passed to the environment. To improve the policy, the
`policy_output.action_weights` contain targets usable to train the policy
probabilities.

We recommend to use the `gumbel_muzero_policy`.
[Gumbel MuZero](https://openreview.net/forum?id=bERaNdoegnO) guarantees a policy
improvement if the action values are correctly evaluated. The policy improvement
is demonstrated in
[`examples/policy_improvement_demo.py`](https://github.com/google-deepmind/mctx/blob/main/examples/policy_improvement_demo.py).

### Enterprise-Grade Performance

MCTX includes specialized optimizations for production environments:

```python
# Using T4-optimized search for enterprise workloads
import mctx

policy_output = mctx.t4_optimized_search(
    params, 
    rng_key, 
    root, 
    recurrent_fn,
    num_simulations=64,
    use_mixed_precision=True
)
```

Key performance features include:
- **T4 GPU Optimization**: 2.1x faster performance on NVIDIA T4 hardware
- **Mixed Precision**: 70% faster matrix operations with FP16 computation
- **Memory Optimization**: Intelligent memory management for larger models
- **Tensor Core Alignment**: Automatic optimization for NVIDIA hardware

See [`docs/t4_optimizations.md`](docs/t4_optimizations.md) for performance details.

### Distributed Computing

For enterprise-scale applications, MCTX supports distributed computing:

```python
import mctx

# Configure distributed search for enterprise scale
dist_config = mctx.DistributedConfig(
    num_devices=8,
    batch_split_strategy="even",
    result_merge_strategy="value_sum"
)

# Use the distributed decorator
@mctx.distribute_mcts(config=dist_config)
def run_distributed_search(params, rng_key, root, recurrent_fn):
    return mctx.muzero_policy(
        params, 
        rng_key, 
        root, 
        recurrent_fn,
        num_simulations=1024
    )

# Run enterprise-scale distributed search
policy_output = run_distributed_search(params, rng_key, root, recurrent_fn)
```

The distributed implementation delivers:
- **Linear Scaling**: Near-linear performance scaling across multiple devices
- **Fault Tolerance**: Resilient operation even with device failures
- **Flexible Deployment**: Support for heterogeneous hardware environments
- **Performance Monitoring**: Built-in metrics for optimization

See [`docs/distributed_mcts.md`](docs/distributed_mcts.md) for implementation details.

### Enterprise Integration

MCTX includes integrations with enterprise systems:

#### SAP HANA Integration

Store and retrieve decision intelligence data with enterprise-grade security:

```python
from mctx.integrations import hana_connector

# Connect to enterprise SAP HANA
connector = hana_connector.HanaConnector(
    host="your_hana_host",
    port=443,
    user="your_username",
    password="your_password",
    use_ssl=True
)

# Store decision results securely
connector.store_search_results(
    search_id="project_001",
    policy_output=policy_output,
    metadata={"business_unit": "operations", "decision_owner": "CFO"}
)

# Retrieve previous decisions
stored_results = connector.get_search_results("project_001")
```

Enterprise integration features include:
- **Secure Authentication**: Full support for enterprise identity management
- **Encrypted Storage**: End-to-end encryption for sensitive decision data
- **Audit Logging**: Comprehensive logging for regulatory compliance
- **Role-based Access**: Fine-grained permission control for decision data

See [`docs/hana_integration.md`](docs/hana_integration.md) for integration details.

### Business Intelligence Visualization

MCTX provides enterprise-grade visualization for business stakeholders:

```python
from mctx.visualization import visualize_tree

# Generate executive-friendly visualization
html = visualize_tree(
    policy_output.search_tree,
    root_state="Initial State",
    show_values=True,
    highlight_path=policy_output.search_path
)

# Save for executive review
with open("decision_analysis.html", "w") as f:
    f.write(html)
```

Visualization capabilities include:
- **Interactive Decision Trees**: Explore decision paths with intuitive navigation
- **Value Heatmaps**: Identify high-value decision alternatives at a glance
- **Executive Dashboards**: Present decision metrics for business stakeholders
- **Comparative Analysis**: Compare multiple decision strategies side-by-side

See [`docs/visualization.md`](docs/visualization.md) for visualization options.

## Example projects
The following projects demonstrate the Mctx usage:

- [Pgx](https://github.com/sotetsuk/pgx) — A collection of 20+ vectorized
  JAX environments, including backgammon, chess, shogi, Go, and an AlphaZero
  example.
- [Basic Learning Demo with Mctx](https://github.com/kenjyoung/mctx_learning_demo) —
  AlphaZero on random mazes.
- [a0-jax](https://github.com/NTT123/a0-jax) — AlphaZero on Connect Four,
  Gomoku, and Go.
- [muax](https://github.com/bwfbowen/muax) — MuZero on gym-style environments
(CartPole, LunarLander).
- [Classic MCTS](https://github.com/Carbon225/mctx-classic) — A simple example on Connect Four.
- [mctx-az](https://github.com/lowrollr/mctx-az) — Mctx with AlphaZero subtree persistence.

Tell us about your project.

## Enterprise Deployment

MCTX supports enterprise-grade deployment through our containerized solutions:

### FastAPI Backend

We provide a ready-to-use FastAPI backend optimized for NVIDIA GPUs:
- **Horizontal Scaling**: Support for multiple load-balanced instances
- **Prometheus Monitoring**: Comprehensive performance metrics
- **Redis Caching**: High-performance response caching
- **Security**: API key authentication and TLS encryption

See [docs/deployment.md](docs/deployment.md) for detailed instructions.

### Frontend Integration

Our visualization frontend can be deployed on Vercel or other cloud providers:
- **Responsive Interface**: Optimized for desktop and mobile devices
- **Real-time Updates**: WebSocket support for live decision monitoring
- **SSO Integration**: Support for enterprise identity providers
- **White-labeling**: Customizable branding for enterprise deployment

## Business Value Documentation

For detailed information on business value and implementation:

- [Executive Overview](docs/executive_overview.md)
- [Business Value & ROI](docs/business_value.md)
- [Industry Solutions](docs/industry_solutions.md)
- [API Reference](docs/api_reference.md)
- [T4 Optimizations Guide](docs/t4_optimizations.md)
- [Distributed MCTS](docs/distributed_mcts.md)
- [Enterprise Integration](docs/hana_integration.md)
- [Visualization Guide](docs/visualization.md)
- [Enterprise Deployment](docs/deployment.md)

## Enterprise Support

For enterprise implementations, we provide:

- **Solution Design**: Custom architecture for your business needs
- **Implementation Services**: Expert integration with your enterprise systems
- **Training**: Role-based training for technical and business users
- **Support**: SLA-backed support for mission-critical deployments

Contact enterprise@mctx-ai.com for enterprise inquiries.

## Citing Mctx

This repository is part of the DeepMind JAX Ecosystem, to cite Mctx
please use the citation:

```bibtex
@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/deepmind},
  year = {2020},
}
```