static bool adding_first_cpu(void)

{

    CPUState *cpu;

    size_t count = 0;

    CPU_FOREACH(cpu) {

        count++;

        if (count > 1) {

            return false;

        }

    }

    return true;

}
