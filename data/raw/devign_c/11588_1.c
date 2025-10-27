altivec_packIntArrayToCharArray(int *val, uint8_t* dest, int dstW) {

  register int i;

  vector unsigned int altivec_vectorShiftInt19 =

    vec_add(vec_splat_u32(10),vec_splat_u32(9));

  if ((unsigned long)dest % 16) {

    /* badly aligned store, we force store alignement */

    /* and will handle load misalignement on val w/ vec_perm */

    vector unsigned char perm1;

    vector signed int v1;

    for (i = 0 ; (i < dstW) &&

	   (((unsigned long)dest + i) % 16) ; i++) {

      int t = val[i] >> 19;

      dest[i] = (t < 0) ? 0 : ((t > 255) ? 255 : t);

    }

    perm1 = vec_lvsl(i << 2, val);

    v1 = vec_ld(i << 2, val);

    for ( ; i < (dstW - 15); i+=16) {

      int offset = i << 2;

      vector signed int v2 = vec_ld(offset + 16, val);

      vector signed int v3 = vec_ld(offset + 32, val);

      vector signed int v4 = vec_ld(offset + 48, val);

      vector signed int v5 = vec_ld(offset + 64, val);

      vector signed int v12 = vec_perm(v1,v2,perm1);

      vector signed int v23 = vec_perm(v2,v3,perm1);

      vector signed int v34 = vec_perm(v3,v4,perm1);

      vector signed int v45 = vec_perm(v4,v5,perm1);



      vector signed int vA = vec_sra(v12, altivec_vectorShiftInt19);

      vector signed int vB = vec_sra(v23, altivec_vectorShiftInt19);

      vector signed int vC = vec_sra(v34, altivec_vectorShiftInt19);

      vector signed int vD = vec_sra(v45, altivec_vectorShiftInt19);

      vector unsigned short vs1 = vec_packsu(vA, vB);

      vector unsigned short vs2 = vec_packsu(vC, vD);

      vector unsigned char vf = vec_packsu(vs1, vs2);

      vec_st(vf, i, dest);

      v1 = v5;

    }

  } else { // dest is properly aligned, great

    for (i = 0; i < (dstW - 15); i+=16) {

      int offset = i << 2;

      vector signed int v1 = vec_ld(offset, val);

      vector signed int v2 = vec_ld(offset + 16, val);

      vector signed int v3 = vec_ld(offset + 32, val);

      vector signed int v4 = vec_ld(offset + 48, val);

      vector signed int v5 = vec_sra(v1, altivec_vectorShiftInt19);

      vector signed int v6 = vec_sra(v2, altivec_vectorShiftInt19);

      vector signed int v7 = vec_sra(v3, altivec_vectorShiftInt19);

      vector signed int v8 = vec_sra(v4, altivec_vectorShiftInt19);

      vector unsigned short vs1 = vec_packsu(v5, v6);

      vector unsigned short vs2 = vec_packsu(v7, v8);

      vector unsigned char vf = vec_packsu(vs1, vs2);

      vec_st(vf, i, dest);

    }

  }

  for ( ; i < dstW ; i++) {

    int t = val[i] >> 19;

    dest[i] = (t < 0) ? 0 : ((t > 255) ? 255 : t);

  }

}
