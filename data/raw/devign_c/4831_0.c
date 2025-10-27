int qemu_input_key_value_to_number(const KeyValue *value)

{

    if (value->kind == KEY_VALUE_KIND_QCODE) {

        return qcode_to_number[value->qcode];

    } else {

        assert(value->kind == KEY_VALUE_KIND_NUMBER);

        return value->number;

    }

}
