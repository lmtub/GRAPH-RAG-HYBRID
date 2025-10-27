static void patch_instruction(VAPICROMState *s, CPUX86State *env, target_ulong ip)

{

    target_phys_addr_t paddr;

    VAPICHandlers *handlers;

    uint8_t opcode[2];

    uint32_t imm32;



    if (smp_cpus == 1) {

        handlers = &s->rom_state.up;

    } else {

        handlers = &s->rom_state.mp;

    }



    pause_all_vcpus();



    cpu_memory_rw_debug(env, ip, opcode, sizeof(opcode), 0);



    switch (opcode[0]) {

    case 0x89: /* mov r32 to r/m32 */

        patch_byte(env, ip, 0x50 + modrm_reg(opcode[1]));  /* push reg */

        patch_call(s, env, ip + 1, handlers->set_tpr);

        break;

    case 0x8b: /* mov r/m32 to r32 */

        patch_byte(env, ip, 0x90);

        patch_call(s, env, ip + 1, handlers->get_tpr[modrm_reg(opcode[1])]);

        break;

    case 0xa1: /* mov abs to eax */

        patch_call(s, env, ip, handlers->get_tpr[0]);

        break;

    case 0xa3: /* mov eax to abs */

        patch_call(s, env, ip, handlers->set_tpr_eax);

        break;

    case 0xc7: /* mov imm32, r/m32 (c7/0) */

        patch_byte(env, ip, 0x68);  /* push imm32 */

        cpu_memory_rw_debug(env, ip + 6, (void *)&imm32, sizeof(imm32), 0);

        cpu_memory_rw_debug(env, ip + 1, (void *)&imm32, sizeof(imm32), 1);

        patch_call(s, env, ip + 5, handlers->set_tpr);

        break;

    case 0xff: /* push r/m32 */

        patch_byte(env, ip, 0x50); /* push eax */

        patch_call(s, env, ip + 1, handlers->get_tpr_stack);

        break;

    default:

        abort();

    }



    resume_all_vcpus();



    paddr = cpu_get_phys_page_debug(env, ip);

    paddr += ip & ~TARGET_PAGE_MASK;

    tb_invalidate_phys_page_range(paddr, paddr + 1, 1);

}
