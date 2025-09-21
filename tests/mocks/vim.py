import os

dirname = os.path.dirname(__file__)

_CODEX_DEFAULTS = {
    "model": "code-davinci-002",
    "endpoint_url": "https://api.openai.com/v1/completions",
    "suffix": "",
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": "",
    "n": 1,
    "stream": 1,
    "logprobs": 0,
    "stop": "",
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "best_of": 1,
    "logit_bias": "",
    "request_timeout": 20,
    "auth_type": "bearer",
    "token_file_path": "",
    "token_load_fn": "",
}

def eval(cmd):
    match cmd:
        case 'g:vim_ai_debug_log_file':
            return '/tmp/vim_ai_debug.log'
        case 'g:vim_ai_roles_config_file':
            return os.path.join(dirname, '../resources/roles.ini')
        case 'g:vim_ai_openai_codex_complete':
            return dict(_CODEX_DEFAULTS)
        case 'g:vim_ai_openai_codex_edit':
            return dict(_CODEX_DEFAULTS)
        case 's:plugin_root':
            return os.path.abspath(os.path.join(dirname, '../..'))
        case 'getcwd()':
            return os.path.abspath(os.path.join(dirname, '../..'))
        case 'g:LoadToken()':
            return 'fn.secret'
        case _:
            return None

def command(cmd):
    pass
