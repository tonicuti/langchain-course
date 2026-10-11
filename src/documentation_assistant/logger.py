# Color codes for better logging
class Colors:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    
def log_info(message: str, color: str = Colors.CYAN):
    """Log infor message with color."""
    print(f"{color}[INFO] {message}{Colors.END}")
    
def log_success(message: str):
    """Log success message in green color"""
    print(f"{Colors.GREEN}[SUCCESS] {message}{Colors.END}")
    
def log_error(message: str):
    """Log error message in red color"""
    print(f"{Colors.RED}[ERROR] {message}{Colors.END}")
    
def log_warning(message: str):
    """Log warning message in yellow color"""
    print(f"{Colors.YELLOW}[WARNING] {message}{Colors.END}")
    
def log_header(message: str):
    """Log header message with emphasis """
    print(f"{Colors.BOLD}{Colors.PURPLE}[HEADER] {message}{Colors.END}")
    
