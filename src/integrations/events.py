from collections.abc import Iterator

from confluent_kafka import Consumer, KafkaError, Producer

from src.common.models import EventEnvelope


class ConfluentEventBus:
    """Uses Confluent Cloud only for lightweight coordination events."""

    def __init__(self, bootstrap_servers: str, api_key: str, api_secret: str) -> None:
        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "security.protocol": "SASL_SSL",
            "sasl.mechanism": "PLAIN",
            "sasl.username": api_key,
            "sasl.password": api_secret,
        }
        self.producer = Producer(self.config)

    def publish(self, topic: str, event: EventEnvelope, timeout: float = 10) -> None:
        delivery_errors: list[KafkaError] = []

        def on_delivery(error: KafkaError | None, _message: object) -> None:
            if error is not None:
                delivery_errors.append(error)

        self.producer.produce(
            topic,
            key=str(event.run_id),
            value=event.model_dump_json(),
            callback=on_delivery,
        )
        remaining = self.producer.flush(timeout)
        if delivery_errors:
            raise RuntimeError(f"Confluent publish failed: {delivery_errors[0]}")
        if remaining:
            raise TimeoutError(f"Confluent publish timed out with {remaining} message(s) pending")

    def consume(self, topic: str, group_id: str) -> Iterator[EventEnvelope]:
        consumer = Consumer({**self.config, "group.id": group_id, "auto.offset.reset": "earliest"})
        consumer.subscribe([topic])
        try:
            while True:
                message = consumer.poll(1)
                if message is None or message.error():
                    continue
                yield EventEnvelope.model_validate_json(message.value())
        finally:
            consumer.close()
