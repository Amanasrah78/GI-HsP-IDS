import pytest

from hsp.expanded.attacks.driver import (
    FAMILIES,
    build_nmap_command,
    classify_mosquitto_rejection,
    reason_code_value,
    validate_request,
)


def test_four_families_are_declared():
    assert set(FAMILIES) == {
        "nmap_connect",
        "python_socket_scan",
        "mosquitto_invalid_auth",
        "paho_invalid_auth",
    }


def test_each_goal_has_two_families():
    goals = [
        definition["attack_goal"]
        for definition in FAMILIES.values()
    ]

    assert goals.count("reconnaissance") == 2
    assert goals.count("authentication") == 2


def test_nmap_command_is_fixed_to_tcp_connect():
    assert build_nmap_command(
        ["172.30.0.10", "172.30.0.11"],
        1883,
    ) == [
        "nmap",
        "-sT",
        "-Pn",
        "-p",
        "1883",
        "172.30.0.10",
        "172.30.0.11",
    ]


@pytest.mark.parametrize(
    "message",
    [
        "Connection Refused: not authorised.",
        "Connection refused: not authorized.",
        "Bad user name or password",
    ],
)
def test_authentication_rejection_is_recognized(message):
    assert classify_mosquitto_rejection(message)


def test_unrelated_failure_is_not_authentication_rejection():
    assert not classify_mosquitto_rejection(
        "Network is unreachable"
    )


def test_request_rejects_external_target():
    with pytest.raises(ValueError, match="allow-list"):
        validate_request(
            "nmap_connect",
            ["8.8.8.8"],
            1883,
            5.0,
            1.0,
        )


def test_request_rejects_non_mqtt_port():
    with pytest.raises(ValueError, match="port 1883"):
        validate_request(
            "nmap_connect",
            ["172.30.0.10"],
            22,
            5.0,
            1.0,
        )


def test_authentication_target_is_restricted():
    with pytest.raises(ValueError, match="172.30.0.12"):
        validate_request(
            "paho_invalid_auth",
            ["172.30.0.10"],
            1883,
            5.0,
            1.0,
        )


def test_reason_code_object_value_is_supported():
    class Reason:
        value = 135

    assert reason_code_value(Reason()) == 135
