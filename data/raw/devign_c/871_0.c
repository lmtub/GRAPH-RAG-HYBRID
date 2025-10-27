Aml *aml_shiftleft(Aml *arg1, Aml *count)

{

    Aml *var = aml_opcode(0x79 /* ShiftLeftOp */);

    aml_append(var, arg1);

    aml_append(var, count);

    build_append_byte(var->buf, 0x00); /* NullNameOp */

    return var;

}
