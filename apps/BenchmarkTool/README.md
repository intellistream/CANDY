<!-- ABOUT THE PROJECT -->
## About BenchmarkTool

The BenchmarkTool is to evaluate the performance of various ANNS algorithms across different scenarios. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Supported Index Type 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Supported Scenarios

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

1. Ensure you have built the whole project before you start.
```sh
cd path_to_the_CANDY/
cmake .
make -j(nproc)
```

2. Download the datasets via scripts.
```sh
cd datasets/
python xxx
```

3. Build you own scenario config file in toml. That is an example for multi-query and multi-insert scenario.

```toml
scenrio = "multi_query_insert"
index_type = "hnsw"
query_thread_count = 10
insert_thread_count = 10
timeout_in_sec = 10
```

4. Run the binary file benchmarl_tooland specify your scenario configuration file path.
```sh
./benchmark_tool config/hnsw.toml
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>