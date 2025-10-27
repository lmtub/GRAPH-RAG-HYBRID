int has_altivec(void)

{

#ifdef __AMIGAOS4__

    ULONG result = 0;

    extern struct ExecIFace *IExec;



    IExec->GetCPUInfoTags(GCIT_VectorUnit, &result, TAG_DONE);

    if (result == VECTORTYPE_ALTIVEC) return 1;


#elif __APPLE__

    int sels[2] = {CTL_HW, HW_VECTORUNIT};

    int has_vu = 0;

    size_t len = sizeof(has_vu);

    int err;



    err = sysctl(sels, 2, &has_vu, &len, NULL, 0);



    if (err == 0) return (has_vu != 0);














#else

    // since we were compiled for altivec, just assume we have it

    // until someone comes up with a proper way (not involving signal hacks).


#endif /* __AMIGAOS4__ */

}