static CCW1 copy_ccw_from_guest(hwaddr addr, bool fmt1)

{

    CCW0 tmp0;

    CCW1 tmp1;

    CCW1 ret;



    if (fmt1) {

        cpu_physical_memory_read(addr, &tmp1, sizeof(tmp1));

        ret.cmd_code = tmp1.cmd_code;

        ret.flags = tmp1.flags;

        ret.count = be16_to_cpu(tmp1.count);

        ret.cda = be32_to_cpu(tmp1.cda);

    } else {

        cpu_physical_memory_read(addr, &tmp0, sizeof(tmp0));

        ret.cmd_code = tmp0.cmd_code;

        ret.flags = tmp0.flags;

        ret.count = be16_to_cpu(tmp0.count);

        ret.cda = be16_to_cpu(tmp0.cda1) | (tmp0.cda0 << 16);

        if ((ret.cmd_code & 0x0f) == CCW_CMD_TIC) {

            ret.cmd_code &= 0x0f;

        }

    }

    return ret;

}
