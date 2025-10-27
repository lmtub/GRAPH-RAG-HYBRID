uint64_t helper_cmpbge(uint64_t op1, uint64_t op2)

{

    uint8_t opa, opb, res;

    int i;



    res = 0;

    for (i = 0; i < 8; i++) {

        opa = op1 >> (i * 8);

        opb = op2 >> (i * 8);

        if (opa >= opb) {

            res |= 1 << i;

        }

    }

    return res;

}
