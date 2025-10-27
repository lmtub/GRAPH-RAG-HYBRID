Aml *aml_or(Aml *arg1, Aml *arg2)

{

    Aml *var = aml_opcode(0x7D /* OrOp */);

    aml_append(var, arg1);

    aml_append(var, arg2);

    build_append_byte(var->buf, 0x00 /* NullNameOp */);

    return var;

}
