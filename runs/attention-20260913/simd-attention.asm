0000f4e4 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)>:
    f4e4: e92d 4ff0    	push.w	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
    f4e8: 4605         	mov	r5, r0
    f4ea: ed2d 8b04    	vpush	{d8, d9}
    f4ee: b091         	sub	sp, #0x44
    f4f0: 4618         	mov	r0, r3
    f4f2: f8dd b078    	ldr.w	r11, [sp, #0x78]
    f4f6: e9dd 741f    	ldrd	r7, r4, [sp, #124]
    f4fa: e9cd 120e    	strd	r1, r2, [sp, #56]
    f4fe: 9309         	str	r3, [sp, #0x24]
    f500: 4659         	mov	r1, r11
    f502: f8dd 9084    	ldr.w	r9, [sp, #0x84]
    f506: f07b ffef    	bl	0x8b4e8 <__divsi3>      @ imm = #0x7bfde
    f50a: ee07 7a90    	vmov	s15, r7
    f50e: fb07 fb0b    	mul	r11, r7, r11
    f512: 900a         	str	r0, [sp, #0x28]
    f514: eeb8 0ae7    	vcvt.f32.s32	s0, s15
    f518: eeb5 0a40    	vcmp.f32	s0, #0
    f51c: eef1 fa10    	vmrs	APSR_nzcv, fpscr
    f520: f100 80b2    	bmi.w	0xf688 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x1a4> @ imm = #0x164
    f524: eeb1 8ac0    	vsqrt.f32	s16, s0
    f528: 9b09         	ldr	r3, [sp, #0x24]
    f52a: 2b00         	cmp	r3, #0x0
    f52c: f340 80a7    	ble.w	0xf67e <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x19a> @ imm = #0x14e
    f530: 00a3         	lsls	r3, r4, #0x2
    f532: 1efc         	subs	r4, r7, #0x3
    f534: 930b         	str	r3, [sp, #0x2c]
    f536: 1f3b         	subs	r3, r7, #0x4
    f538: f023 0303    	bic	r3, r3, #0x3
    f53c: 9503         	str	r5, [sp, #0xc]
    f53e: 3304         	adds	r3, #0x4
    f540: 930d         	str	r3, [sp, #0x34]
    f542: e9dd 1322    	ldrd	r1, r3, [sp, #136]
    f546: 00ba         	lsls	r2, r7, #0x2
    f548: 9101         	str	r1, [sp, #0x4]
    f54a: 2100         	movs	r1, #0x0
    f54c: ea4f 0b8b    	lsl.w	r11, r11, #0x2
    f550: 189e         	adds	r6, r3, r2
    f552: 9102         	str	r1, [sp, #0x8]
    f554: f109 0101    	add.w	r1, r9, #0x1
    f558: 910c         	str	r1, [sp, #0x30]
    f55a: 9204         	str	r2, [sp, #0x10]
    f55c: 990a         	ldr	r1, [sp, #0x28]
    f55e: 9802         	ldr	r0, [sp, #0x8]
    f560: 9305         	str	r3, [sp, #0x14]
    f562: f07b ffc1    	bl	0x8b4e8 <__divsi3>      @ imm = #0x7bf82
    f566: 9b05         	ldr	r3, [sp, #0x14]
    f568: f1b9 0f00    	cmp.w	r9, #0x0
    f56c: fb07 f000    	mul	r0, r7, r0
    f570: db68         	blt	0xf644 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x160> @ imm = #0xd0
    f572: 990e         	ldr	r1, [sp, #0x38]
    f574: f04f 0800    	mov.w	r8, #0x0
    f578: 0082         	lsls	r2, r0, #0x2
    f57a: f8dd a004    	ldr.w	r10, [sp, #0x4]
    f57e: 188d         	adds	r5, r1, r2
    f580: 9606         	str	r6, [sp, #0x18]
    f582: 4656         	mov	r6, r10
    f584: 9407         	str	r4, [sp, #0x1c]
    f586: 469a         	mov	r10, r3
    f588: 9205         	str	r2, [sp, #0x14]
    f58a: 462c         	mov	r4, r5
    f58c: 9d03         	ldr	r5, [sp, #0xc]
    f58e: 4620         	mov	r0, r4
    f590: 463a         	mov	r2, r7
    f592: 445c         	add	r4, r11
    f594: 4629         	mov	r1, r5
    f596: f108 0801    	add.w	r8, r8, #0x1
    f59a: f7ff fab3    	bl	0xeb04 <unremarkable::(anonymous namespace)::dot_fp32(float const*, float const*, int)> @ imm = #-0xa9a
    f59e: eec0 7a08    	vdiv.f32	s15, s0, s16
    f5a2: 45c1         	cmp	r9, r8
    f5a4: ece6 7a01    	vstmia	r6!, {s15}
    f5a8: daf1         	bge	0xf58e <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xaa> @ imm = #-0x1e
    f5aa: f8dd 8004    	ldr.w	r8, [sp, #0x4]
    f5ae: 990c         	ldr	r1, [sp, #0x30]
    f5b0: f8cd a020    	str.w	r10, [sp, #0x20]
    f5b4: 4640         	mov	r0, r8
    f5b6: e9dd 6406    	ldrd	r6, r4, [sp, #24]
    f5ba: f7ff fb91    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x8de
    f5be: 9b08         	ldr	r3, [sp, #0x20]
    f5c0: 2100         	movs	r1, #0x0
    f5c2: 9a04         	ldr	r2, [sp, #0x10]
    f5c4: 4618         	mov	r0, r3
    f5c6: f7f8 eb30    	blx	0x7c28 <memset@plt>     @ imm = #-0x79a0
    f5ca: 9d05         	ldr	r5, [sp, #0x14]
    f5cc: 2f03         	cmp	r7, #0x3
    f5ce: 9a0f         	ldr	r2, [sp, #0x3c]
    f5d0: 4603         	mov	r3, r0
    f5d2: f8dd a034    	ldr.w	r10, [sp, #0x34]
    f5d6: f04f 0e00    	mov.w	lr, #0x0
    f5da: 4415         	add	r5, r2
    f5dc: f858 cb04    	ldr	r12, [r8], #4
    f5e0: dd2e         	ble	0xf640 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x15c> @ imm = #0x5c
    f5e2: 4619         	mov	r1, r3
    f5e4: 4628         	mov	r0, r5
    f5e6: 2200         	movs	r2, #0x0
    f5e8: f960 2a8d    	vld1.32	{d18, d19}, [r0]!
    f5ec: ee09 cb10    	vmov.32	d9[0], r12
    f5f0: f961 0a8f    	vld1.32	{d16, d17}, [r1]
    f5f4: ffe2 29c9    	vmul.f32	q9, q9, d9[0]
    f5f8: 3204         	adds	r2, #0x4
    f5fa: 42a2         	cmp	r2, r4
    f5fc: ef40 0de2    	vadd.f32	q8, q8, q9
    f600: f941 0a8d    	vst1.32	{d16, d17}, [r1]!
    f604: dbf0         	blt	0xf5e8 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x104> @ imm = #-0x20
    f606: 4651         	mov	r1, r10
    f608: 428f         	cmp	r7, r1
    f60a: dd0f         	ble	0xf62c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x148> @ imm = #0x1e
    f60c: 0089         	lsls	r1, r1, #0x2
    f60e: 4660         	mov	r0, r12
    f610: 185a         	adds	r2, r3, r1
    f612: 4429         	add	r1, r5
    f614: ee06 0a90    	vmov	s13, r0
    f618: edd2 7a00    	vldr	s15, [r2]
    f61c: ecb1 7a01    	vldmia	r1!, {s14}
    f620: ee47 7a26    	vmla.f32	s15, s14, s13
    f624: ece2 7a01    	vstmia	r2!, {s15}
    f628: 42b2         	cmp	r2, r6
    f62a: d1f3         	bne	0xf614 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x130> @ imm = #-0x1a
    f62c: 45f1         	cmp	r9, lr
    f62e: f10e 0201    	add.w	r2, lr, #0x1
    f632: 445d         	add	r5, r11
    f634: d012         	beq	0xf65c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x178> @ imm = #0x24
    f636: f858 cb04    	ldr	r12, [r8], #4
    f63a: 2f03         	cmp	r7, #0x3
    f63c: 4696         	mov	lr, r2
    f63e: dcd0         	bgt	0xf5e2 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0xfe> @ imm = #-0x60
    f640: 2100         	movs	r1, #0x0
    f642: e7e1         	b	0xf608 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x124> @ imm = #-0x3e
    f644: 990c         	ldr	r1, [sp, #0x30]
    f646: 9801         	ldr	r0, [sp, #0x4]
    f648: 9305         	str	r3, [sp, #0x14]
    f64a: f7ff fb49    	bl	0xece0 <unremarkable::softmax_inplace(float*, int)> @ imm = #-0x96e
    f64e: 9b05         	ldr	r3, [sp, #0x14]
    f650: 2100         	movs	r1, #0x0
    f652: 9a04         	ldr	r2, [sp, #0x10]
    f654: 4618         	mov	r0, r3
    f656: f7f8 eae8    	blx	0x7c28 <memset@plt>     @ imm = #-0x7a30
    f65a: 4603         	mov	r3, r0
    f65c: 9a01         	ldr	r2, [sp, #0x4]
    f65e: 990b         	ldr	r1, [sp, #0x2c]
    f660: 9803         	ldr	r0, [sp, #0xc]
    f662: 440a         	add	r2, r1
    f664: 9904         	ldr	r1, [sp, #0x10]
    f666: 9201         	str	r2, [sp, #0x4]
    f668: 9a02         	ldr	r2, [sp, #0x8]
    f66a: 4408         	add	r0, r1
    f66c: 440b         	add	r3, r1
    f66e: 3201         	adds	r2, #0x1
    f670: 440e         	add	r6, r1
    f672: 9909         	ldr	r1, [sp, #0x24]
    f674: 9202         	str	r2, [sp, #0x8]
    f676: 4291         	cmp	r1, r2
    f678: 9003         	str	r0, [sp, #0xc]
    f67a: f47f af6f    	bne.w	0xf55c <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x78> @ imm = #-0x122
    f67e: b011         	add	sp, #0x44
    f680: ecbd 8b04    	vpop	{d8, d9}
    f684: e8bd 8ff0    	pop.w	{r4, r5, r6, r7, r8, r9, r10, r11, pc}
    f688: f7f8 ea72    	blx	0x7b70 <sqrtf@plt>      @ imm = #-0x7b1c
    f68c: eeb0 8a40    	vmov.f32	s16, s0
    f690: e74a         	b	0xf528 <unremarkable::causal_attention(float const*, float const*, float const*, int, int, int, int, int, float*, float*)+0x44> @ imm = #-0x16c
    f692: bf00         	nop

