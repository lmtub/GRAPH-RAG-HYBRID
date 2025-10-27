int qemu_input_key_value_to_qcode(const KeyValue *value)

{

    if (value->type == KEY_VALUE_KIND_QCODE) {

        return value->u.qcode;

    } else {

        assert(value->type == KEY_VALUE_KIND_NUMBER);

        return qemu_input_key_number_to_qcode(value->u.number);

    }

}
