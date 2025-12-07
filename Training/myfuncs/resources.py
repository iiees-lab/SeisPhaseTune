import psutil
import cpuinfo
import logging
import torch
import subprocess

def get_cpu_info():
    """
    Retrieves a formatted summary of the system's CPU information.

    Uses the `cpuinfo` and `psutil` libraries to extract:
    - CPU model/brand name
    - Number of logical (hyper-threaded) cores
    - Number of physical cores

    Returns
    -------
    str
        A formatted multi-line string containing the CPU model name,
        number of logical cores, and number of physical cores.

    Notes
    -----
    - Requires the `py-cpuinfo` and `psutil` libraries.
    - On some systems, the CPU model name may be returned as "Unknown CPU"
      if not detected properly.
    """
    # CPU model name
    info = cpuinfo.get_cpu_info()
    cpu_model = info.get('brand_raw', 'Unknown CPU')
    # Number of logical and physical cores
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)
    txt = (
         "\n"
        f"{'CPU Model:':25} {cpu_model}\n"
        f"{'Logical cores (threads):':25} {logical_cores}\n"
        f"{'Physical cores:':25} {physical_cores}\n"
    )
    return txt


def Check_CUDA_availability():
    logging.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)

        logging.info("CUDA device count: %d", torch.cuda.device_count())
        logging.info("Current device index: %d", idx)
        logging.info("Current device name: %s", torch.cuda.get_device_name(idx))

        logging.info("\n=== Device Properties ===")
        logging.info("Name: %s", props.name)
        logging.info("Total memory (GB): %.2f", props.total_memory / 1024**3)
        logging.info("Multiprocessors: %d", props.multi_processor_count)
        logging.info("Compute capability: %d.%d", props.major, props.minor)

        # Safe getter for version-dependent properties
        def safe(prop):
            return getattr(props, prop, "N/A")

        logging.info("Max threads per block: %s", safe("max_threads_per_block"))
        logging.info("Max threads per multiprocessor: %s", safe("max_threads_per_multi_processor"))
        logging.info("Shared memory per block (bytes): %s", safe("shared_memory_per_block"))
        logging.info("Warp size: %s", safe("warp_size"))
        logging.info("Clock rate (kHz): %s", safe("clock_rate"))

        logging.info("\n=== Memory Usage ===")
        logging.info("Allocated (GB): %.3f", torch.cuda.memory_allocated() / 1024**3)
        logging.info("Reserved (GB): %.3f", torch.cuda.memory_reserved() / 1024**3)

        # Optional: nvidia-smi output
        try:
            logging.info("\n=== nvidia-smi Output ===")
            smi_output = subprocess.check_output(["nvidia-smi"], encoding="utf-8")
            logging.info("\n%s", smi_output)
        except Exception as e:
            logging.info("nvidia-smi not available: %s", e)

    else:
        logging.info("CUDA is not available on this environment.")