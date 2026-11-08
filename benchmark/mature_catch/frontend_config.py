"""Registered benchmark frontend settings. Never changes generator/core settings."""

KEYS = {
    'CATCHIT_EVENT_OBSERVATION_WINDOW_MS': ('0', '10'),
    'CATCHIT_EVENT_FLIGHT_FIT': ('0', '1'),
    'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS': ('50', '100'),
}


def validate_frontend(values):
    if set(values) != set(KEYS):
        raise ValueError('Every selected frontend must declare exactly the three registered settings')
    result = {k: str(v) for k, v in values.items()}
    if any(result[k] not in choices for k, choices in KEYS.items()):
        raise ValueError('Unregistered frontend setting')
    return result


def resolve_frontend(spec, case, method):
    """Shared crossover rows override method-selected settings; default is unchanged."""
    selected = spec.get('frontend_by_method', {}).get(method)
    if 'frontend_overrides' in case:
        selected = case['frontend_overrides']
    return {} if selected is None else validate_frontend(selected)


def candidate_grid():
    return [dict(frontend_id=f'w{w}_fit{f}_gap{g}', env=validate_frontend({
        'CATCHIT_EVENT_OBSERVATION_WINDOW_MS': str(w),
        'CATCHIT_EVENT_FLIGHT_FIT': str(f),
        'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS': str(g),
    })) for w in (0, 10) for f in (0, 1) for g in (50, 100)]
