static void OPLWriteReg(FM_OPL *OPL, int r, int v)

{

	OPL_CH *CH;

	int slot;

	int block_fnum;



	switch(r&0xe0)

	{

	case 0x00: /* 00-1f:control */

		switch(r&0x1f)

		{

		case 0x01:

			/* wave selector enable */

			if(OPL->type&OPL_TYPE_WAVESEL)

			{

				OPL->wavesel = v&0x20;

				if(!OPL->wavesel)

				{

					/* preset compatible mode */

					int c;

					for(c=0;c<OPL->max_ch;c++)

					{

						OPL->P_CH[c].SLOT[SLOT1].wavetable = &SIN_TABLE[0];

						OPL->P_CH[c].SLOT[SLOT2].wavetable = &SIN_TABLE[0];

					}

				}

			}

			return;

		case 0x02:	/* Timer 1 */

			OPL->T[0] = (256-v)*4;

			break;

		case 0x03:	/* Timer 2 */

			OPL->T[1] = (256-v)*16;

			return;

		case 0x04:	/* IRQ clear / mask and Timer enable */

			if(v&0x80)

			{	/* IRQ flag clear */

				OPL_STATUS_RESET(OPL,0x7f);

			}

			else

			{	/* set IRQ mask ,timer enable*/

				UINT8 st1 = v&1;

				UINT8 st2 = (v>>1)&1;

				/* IRQRST,T1MSK,t2MSK,EOSMSK,BRMSK,x,ST2,ST1 */

				OPL_STATUS_RESET(OPL,v&0x78);

				OPL_STATUSMASK_SET(OPL,((~v)&0x78)|0x01);

				/* timer 2 */

				if(OPL->st[1] != st2)

				{

					double interval = st2 ? (double)OPL->T[1]*OPL->TimerBase : 0.0;

					OPL->st[1] = st2;

					if (OPL->TimerHandler) (OPL->TimerHandler)(OPL->TimerParam+1,interval);

				}

				/* timer 1 */

				if(OPL->st[0] != st1)

				{

					double interval = st1 ? (double)OPL->T[0]*OPL->TimerBase : 0.0;

					OPL->st[0] = st1;

					if (OPL->TimerHandler) (OPL->TimerHandler)(OPL->TimerParam+0,interval);

				}

			}

			return;

#if BUILD_Y8950

		case 0x06:		/* Key Board OUT */

			if(OPL->type&OPL_TYPE_KEYBOARD)

			{

				if(OPL->keyboardhandler_w)

					OPL->keyboardhandler_w(OPL->keyboard_param,v);

				else

					LOG(LOG_WAR,("OPL:write unmapped KEYBOARD port\n"));

			}

			return;

		case 0x07:	/* DELTA-T control : START,REC,MEMDATA,REPT,SPOFF,x,x,RST */

			if(OPL->type&OPL_TYPE_ADPCM)

				YM_DELTAT_ADPCM_Write(OPL->deltat,r-0x07,v);

			return;

		case 0x08:	/* MODE,DELTA-T : CSM,NOTESEL,x,x,smpl,da/ad,64k,rom */

			OPL->mode = v;

			v&=0x1f;	/* for DELTA-T unit */

		case 0x09:		/* START ADD */

		case 0x0a:

		case 0x0b:		/* STOP ADD  */

		case 0x0c:

		case 0x0d:		/* PRESCALE   */

		case 0x0e:

		case 0x0f:		/* ADPCM data */

		case 0x10: 		/* DELTA-N    */

		case 0x11: 		/* DELTA-N    */

		case 0x12: 		/* EG-CTRL    */

			if(OPL->type&OPL_TYPE_ADPCM)

				YM_DELTAT_ADPCM_Write(OPL->deltat,r-0x07,v);

			return;

#if 0

		case 0x15:		/* DAC data    */

		case 0x16:

		case 0x17:		/* SHIFT    */

			return;

		case 0x18:		/* I/O CTRL (Direction) */

			if(OPL->type&OPL_TYPE_IO)

				OPL->portDirection = v&0x0f;

			return;

		case 0x19:		/* I/O DATA */

			if(OPL->type&OPL_TYPE_IO)

			{

				OPL->portLatch = v;

				if(OPL->porthandler_w)

					OPL->porthandler_w(OPL->port_param,v&OPL->portDirection);

			}

			return;

		case 0x1a:		/* PCM data */

			return;

#endif

#endif

		}

		break;

	case 0x20:	/* am,vib,ksr,eg type,mul */

		slot = slot_array[r&0x1f];

		if(slot == -1) return;

		set_mul(OPL,slot,v);

		return;

	case 0x40:

		slot = slot_array[r&0x1f];

		if(slot == -1) return;

		set_ksl_tl(OPL,slot,v);

		return;

	case 0x60:

		slot = slot_array[r&0x1f];

		if(slot == -1) return;

		set_ar_dr(OPL,slot,v);

		return;

	case 0x80:

		slot = slot_array[r&0x1f];

		if(slot == -1) return;

		set_sl_rr(OPL,slot,v);

		return;

	case 0xa0:

		switch(r)

		{

		case 0xbd:

			/* amsep,vibdep,r,bd,sd,tom,tc,hh */

			{

			UINT8 rkey = OPL->rythm^v;

			OPL->ams_table = &AMS_TABLE[v&0x80 ? AMS_ENT : 0];

			OPL->vib_table = &VIB_TABLE[v&0x40 ? VIB_ENT : 0];

			OPL->rythm  = v&0x3f;

			if(OPL->rythm&0x20)

			{

#if 0

				usrintf_showmessage("OPL Rythm mode select");

#endif

				/* BD key on/off */

				if(rkey&0x10)

				{

					if(v&0x10)

					{

						OPL->P_CH[6].op1_out[0] = OPL->P_CH[6].op1_out[1] = 0;

						OPL_KEYON(&OPL->P_CH[6].SLOT[SLOT1]);

						OPL_KEYON(&OPL->P_CH[6].SLOT[SLOT2]);

					}

					else

					{

						OPL_KEYOFF(&OPL->P_CH[6].SLOT[SLOT1]);

						OPL_KEYOFF(&OPL->P_CH[6].SLOT[SLOT2]);

					}

				}

				/* SD key on/off */

				if(rkey&0x08)

				{

					if(v&0x08) OPL_KEYON(&OPL->P_CH[7].SLOT[SLOT2]);

					else       OPL_KEYOFF(&OPL->P_CH[7].SLOT[SLOT2]);

				}/* TAM key on/off */

				if(rkey&0x04)

				{

					if(v&0x04) OPL_KEYON(&OPL->P_CH[8].SLOT[SLOT1]);

					else       OPL_KEYOFF(&OPL->P_CH[8].SLOT[SLOT1]);

				}

				/* TOP-CY key on/off */

				if(rkey&0x02)

				{

					if(v&0x02) OPL_KEYON(&OPL->P_CH[8].SLOT[SLOT2]);

					else       OPL_KEYOFF(&OPL->P_CH[8].SLOT[SLOT2]);

				}

				/* HH key on/off */

				if(rkey&0x01)

				{

					if(v&0x01) OPL_KEYON(&OPL->P_CH[7].SLOT[SLOT1]);

					else       OPL_KEYOFF(&OPL->P_CH[7].SLOT[SLOT1]);

				}

			}

			}

			return;

		}

		/* keyon,block,fnum */

		if( (r&0x0f) > 8) return;

		CH = &OPL->P_CH[r&0x0f];

		if(!(r&0x10))

		{	/* a0-a8 */

			block_fnum  = (CH->block_fnum&0x1f00) | v;

		}

		else

		{	/* b0-b8 */

			int keyon = (v>>5)&1;

			block_fnum = ((v&0x1f)<<8) | (CH->block_fnum&0xff);

			if(CH->keyon != keyon)

			{

				if( (CH->keyon=keyon) )

				{

					CH->op1_out[0] = CH->op1_out[1] = 0;

					OPL_KEYON(&CH->SLOT[SLOT1]);

					OPL_KEYON(&CH->SLOT[SLOT2]);

				}

				else

				{

					OPL_KEYOFF(&CH->SLOT[SLOT1]);

					OPL_KEYOFF(&CH->SLOT[SLOT2]);

				}

			}

		}

		/* update */

		if(CH->block_fnum != block_fnum)

		{

			int blockRv = 7-(block_fnum>>10);

			int fnum   = block_fnum&0x3ff;

			CH->block_fnum = block_fnum;



			CH->ksl_base = KSL_TABLE[block_fnum>>6];

			CH->fc = OPL->FN_TABLE[fnum]>>blockRv;

			CH->kcode = CH->block_fnum>>9;

			if( (OPL->mode&0x40) && CH->block_fnum&0x100) CH->kcode |=1;

			CALC_FCSLOT(CH,&CH->SLOT[SLOT1]);

			CALC_FCSLOT(CH,&CH->SLOT[SLOT2]);

		}

		return;

	case 0xc0:

		/* FB,C */

		if( (r&0x0f) > 8) return;

		CH = &OPL->P_CH[r&0x0f];

		{

		int feedback = (v>>1)&7;

		CH->FB   = feedback ? (8+1) - feedback : 0;

		CH->CON = v&1;

		set_algorythm(CH);

		}

		return;

	case 0xe0: /* wave type */

		slot = slot_array[r&0x1f];

		if(slot == -1) return;

		CH = &OPL->P_CH[slot/2];

		if(OPL->wavesel)

		{

			/* LOG(LOG_INF,("OPL SLOT %d wave select %d\n",slot,v&3)); */

			CH->SLOT[slot&1].wavetable = &SIN_TABLE[(v&0x03)*SIN_ENT];

		}

		return;

	}

}
