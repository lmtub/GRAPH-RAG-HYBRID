bool kvm_arch_stop_on_emulation_error(CPUState *env)

{

      return !(env->cr[0] & CR0_PE_MASK) ||

              ((env->segs[R_CS].selector  & 3) != 3);

}
