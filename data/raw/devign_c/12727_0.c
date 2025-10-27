Aml *aml_arg(int pos)

{

    Aml *var;

    uint8_t op = 0x68 /* ARG0 op */ + pos;



    assert(pos <= 6);

    var = aml_opcode(op);

    return var;

}
