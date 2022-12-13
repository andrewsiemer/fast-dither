
src/MedianPartition.o:     file format elf64-x86-64


Disassembly of section .text:

0000000000000000 <MedianPartition>:
       0:	55                   	push   %rbp
       1:	41 57                	push   %r15
       3:	41 56                	push   %r14
       5:	41 55                	push   %r13
       7:	41 54                	push   %r12
       9:	53                   	push   %rbx
       a:	48 83 ec 68          	sub    $0x68,%rsp
       e:	48 89 4c 24 38       	mov    %rcx,0x38(%rsp)
      13:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
      18:	48 85 f6             	test   %rsi,%rsi
      1b:	0f 84 6f 02 00 00    	je     290 <MedianPartition+0x290>
      21:	4c 89 4c 24 40       	mov    %r9,0x40(%rsp)
      26:	48 85 d2             	test   %rdx,%rdx
      29:	0f 84 7a 02 00 00    	je     2a9 <MedianPartition+0x2a9>
      2f:	48 83 7c 24 38 00    	cmpq   $0x0,0x38(%rsp)
      35:	0f 84 87 02 00 00    	je     2c2 <MedianPartition+0x2c2>
      3b:	48 89 54 24 28       	mov    %rdx,0x28(%rsp)
      40:	4d 85 c0             	test   %r8,%r8
      43:	48 89 f0             	mov    %rsi,%rax
      46:	0f 84 8f 02 00 00    	je     2db <MedianPartition+0x2db>
      4c:	4c 89 c1             	mov    %r8,%rcx
      4f:	48 d1 e9             	shr    %rcx
      52:	48 89 4c 24 18       	mov    %rcx,0x18(%rsp)
      57:	4c 39 c1             	cmp    %r8,%rcx
      5a:	0f 83 94 02 00 00    	jae    2f4 <MedianPartition+0x2f4>
      60:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
      65:	49 89 c5             	mov    %rax,%r13
      68:	49 83 f8 02          	cmp    $0x2,%r8
      6c:	4c 89 44 24 10       	mov    %r8,0x10(%rsp)
      71:	0f 82 12 01 00 00    	jb     189 <MedianPartition+0x189>
      77:	41 b4 ff             	mov    $0xff,%r12b
      7a:	45 31 f6             	xor    %r14d,%r14d
      7d:	4c 8b 6c 24 30       	mov    0x30(%rsp),%r13
      82:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
      87:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
      8c:	48 8b 44 24 38       	mov    0x38(%rsp),%rax
      91:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
      96:	48 8b 4c 24 10       	mov    0x10(%rsp),%rcx
      9b:	48 8b 5c 24 18       	mov    0x18(%rsp),%rbx
      a0:	44 88 64 24 27       	mov    %r12b,0x27(%rsp)
      a5:	49 89 cf             	mov    %rcx,%r15
      a8:	e8 00 00 00 00       	callq  ad <MedianPartition+0xad>
      ad:	89 c0                	mov    %eax,%eax
      af:	31 d2                	xor    %edx,%edx
      b1:	49 f7 f7             	div    %r15
      b4:	41 8a 6c 15 00       	mov    0x0(%r13,%rdx,1),%bpl
      b9:	44 38 f5             	cmp    %r14b,%bpl
      bc:	0f 82 4b 02 00 00    	jb     30d <MedianPartition+0x30d>
      c2:	48 89 5c 24 48       	mov    %rbx,0x48(%rsp)
      c7:	4c 89 74 24 08       	mov    %r14,0x8(%rsp)
      cc:	44 38 e5             	cmp    %r12b,%bpl
      cf:	72 03                	jb     d4 <MedianPartition+0xd4>
      d1:	44 88 e5             	mov    %r12b,%bpl
      d4:	0f 31                	rdtsc  
      d6:	41 89 d4             	mov    %edx,%r12d
      d9:	89 c3                	mov    %eax,%ebx
      db:	4c 8b 74 24 40       	mov    0x40(%rsp),%r14
      e0:	4c 89 34 24          	mov    %r14,(%rsp)
      e4:	44 0f b6 cd          	movzbl %bpl,%r9d
      e8:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
      ed:	4c 89 ee             	mov    %r13,%rsi
      f0:	48 8b 54 24 58       	mov    0x58(%rsp),%rdx
      f5:	48 8b 4c 24 50       	mov    0x50(%rsp),%rcx
      fa:	4d 89 f8             	mov    %r15,%r8
      fd:	e8 ae 02 00 00       	callq  3b0 <Partition>
     102:	48 89 c1             	mov    %rax,%rcx
     105:	0f 31                	rdtsc  
     107:	49 39 cf             	cmp    %rcx,%r15
     10a:	0f 82 16 02 00 00    	jb     326 <MedianPartition+0x326>
     110:	48 c1 e2 20          	shl    $0x20,%rdx
     114:	89 c0                	mov    %eax,%eax
     116:	49 c1 e4 20          	shl    $0x20,%r12
     11a:	89 de                	mov    %ebx,%esi
     11c:	48 09 d0             	or     %rdx,%rax
     11f:	4c 09 e6             	or     %r12,%rsi
     122:	4d 01 7e 18          	add    %r15,0x18(%r14)
     126:	48 29 f0             	sub    %rsi,%rax
     129:	49 01 46 10          	add    %rax,0x10(%r14)
     12d:	48 8b 5c 24 48       	mov    0x48(%rsp),%rbx
     132:	48 89 d8             	mov    %rbx,%rax
     135:	48 29 c8             	sub    %rcx,%rax
     138:	72 35                	jb     16f <MedianPartition+0x16f>
     13a:	49 29 cf             	sub    %rcx,%r15
     13d:	44 8a 64 24 27       	mov    0x27(%rsp),%r12b
     142:	0f 86 f7 01 00 00    	jbe    33f <MedianPartition+0x33f>
     148:	49 01 cd             	add    %rcx,%r13
     14b:	49 83 ff 02          	cmp    $0x2,%r15
     14f:	72 38                	jb     189 <MedianPartition+0x189>
     151:	48 01 4c 24 58       	add    %rcx,0x58(%rsp)
     156:	48 01 4c 24 50       	add    %rcx,0x50(%rsp)
     15b:	40 fe c5             	inc    %bpl
     15e:	48 89 c3             	mov    %rax,%rbx
     161:	41 88 ee             	mov    %bpl,%r14b
     164:	44 38 e5             	cmp    %r12b,%bpl
     167:	0f 86 3b ff ff ff    	jbe    a8 <MedianPartition+0xa8>
     16d:	eb 1a                	jmp    189 <MedianPartition+0x189>
     16f:	48 83 f9 02          	cmp    $0x2,%rcx
     173:	4c 8b 74 24 08       	mov    0x8(%rsp),%r14
     178:	72 0f                	jb     189 <MedianPartition+0x189>
     17a:	40 fe cd             	dec    %bpl
     17d:	41 88 ec             	mov    %bpl,%r12b
     180:	41 38 ee             	cmp    %bpl,%r14b
     183:	0f 86 17 ff ff ff    	jbe    a0 <MedianPartition+0xa0>
     189:	45 8a 7d 00          	mov    0x0(%r13),%r15b
     18d:	0f 31                	rdtsc  
     18f:	89 d5                	mov    %edx,%ebp
     191:	41 89 c4             	mov    %eax,%r12d
     194:	48 8b 5c 24 40       	mov    0x40(%rsp),%rbx
     199:	48 89 1c 24          	mov    %rbx,(%rsp)
     19d:	45 0f b6 cf          	movzbl %r15b,%r9d
     1a1:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
     1a6:	48 8b 74 24 30       	mov    0x30(%rsp),%rsi
     1ab:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
     1b0:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
     1b5:	4c 8b 6c 24 10       	mov    0x10(%rsp),%r13
     1ba:	4d 89 e8             	mov    %r13,%r8
     1bd:	e8 ee 01 00 00       	callq  3b0 <Partition>
     1c2:	49 89 c6             	mov    %rax,%r14
     1c5:	0f 31                	rdtsc  
     1c7:	48 8b 74 24 18       	mov    0x18(%rsp),%rsi
     1cc:	49 39 f6             	cmp    %rsi,%r14
     1cf:	0f 86 83 01 00 00    	jbe    358 <MedianPartition+0x358>
     1d5:	48 c1 e2 20          	shl    $0x20,%rdx
     1d9:	89 c0                	mov    %eax,%eax
     1db:	48 c1 e5 20          	shl    $0x20,%rbp
     1df:	44 89 e1             	mov    %r12d,%ecx
     1e2:	48 09 d0             	or     %rdx,%rax
     1e5:	48 09 e9             	or     %rbp,%rcx
     1e8:	48 29 c8             	sub    %rcx,%rax
     1eb:	c4 e1 f9 6e c0       	vmovq  %rax,%xmm0
     1f0:	c4 c1 f9 6e cd       	vmovq  %r13,%xmm1
     1f5:	c5 f9 6c c1          	vpunpcklqdq %xmm1,%xmm0,%xmm0
     1f9:	c5 f9 d4 43 10       	vpaddq 0x10(%rbx),%xmm0,%xmm0
     1fe:	c5 fa 7f 43 10       	vmovdqu %xmm0,0x10(%rbx)
     203:	45 84 ff             	test   %r15b,%r15b
     206:	4c 8b 6c 24 30       	mov    0x30(%rsp),%r13
     20b:	74 66                	je     273 <MedianPartition+0x273>
     20d:	44 88 f9             	mov    %r15b,%cl
     210:	fe c9                	dec    %cl
     212:	0f 31                	rdtsc  
     214:	89 d5                	mov    %edx,%ebp
     216:	89 44 24 58          	mov    %eax,0x58(%rsp)
     21a:	48 89 1c 24          	mov    %rbx,(%rsp)
     21e:	44 0f b6 c9          	movzbl %cl,%r9d
     222:	48 8b 7c 24 60       	mov    0x60(%rsp),%rdi
     227:	49 89 f4             	mov    %rsi,%r12
     22a:	4c 89 ee             	mov    %r13,%rsi
     22d:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
     232:	48 8b 4c 24 38       	mov    0x38(%rsp),%rcx
     237:	4d 89 f0             	mov    %r14,%r8
     23a:	e8 71 01 00 00       	callq  3b0 <Partition>
     23f:	4c 89 e6             	mov    %r12,%rsi
     242:	48 39 f0             	cmp    %rsi,%rax
     245:	0f 87 26 01 00 00    	ja     371 <MedianPartition+0x371>
     24b:	48 c1 e5 20          	shl    $0x20,%rbp
     24f:	8b 4c 24 58          	mov    0x58(%rsp),%ecx
     253:	48 09 e9             	or     %rbp,%rcx
     256:	0f 31                	rdtsc  
     258:	89 c0                	mov    %eax,%eax
     25a:	48 c1 e2 20          	shl    $0x20,%rdx
     25e:	48 09 c2             	or     %rax,%rdx
     261:	4c 01 73 18          	add    %r14,0x18(%rbx)
     265:	48 8b 43 10          	mov    0x10(%rbx),%rax
     269:	48 29 c8             	sub    %rcx,%rax
     26c:	48 01 d0             	add    %rdx,%rax
     26f:	48 89 43 10          	mov    %rax,0x10(%rbx)
     273:	45 38 7c 35 00       	cmp    %r15b,0x0(%r13,%rsi,1)
     278:	0f 85 0c 01 00 00    	jne    38a <MedianPartition+0x38a>
     27e:	48 89 f0             	mov    %rsi,%rax
     281:	48 83 c4 68          	add    $0x68,%rsp
     285:	5b                   	pop    %rbx
     286:	41 5c                	pop    %r12
     288:	41 5d                	pop    %r13
     28a:	41 5e                	pop    %r14
     28c:	41 5f                	pop    %r15
     28e:	5d                   	pop    %rbp
     28f:	c3                   	retq   
     290:	bf 00 00 00 00       	mov    $0x0,%edi
     295:	be 00 00 00 00       	mov    $0x0,%esi
     29a:	ba ca 02 00 00       	mov    $0x2ca,%edx
     29f:	b9 00 00 00 00       	mov    $0x0,%ecx
     2a4:	e8 00 00 00 00       	callq  2a9 <MedianPartition+0x2a9>
     2a9:	bf 00 00 00 00       	mov    $0x0,%edi
     2ae:	be 00 00 00 00       	mov    $0x0,%esi
     2b3:	ba cb 02 00 00       	mov    $0x2cb,%edx
     2b8:	b9 00 00 00 00       	mov    $0x0,%ecx
     2bd:	e8 00 00 00 00       	callq  2c2 <MedianPartition+0x2c2>
     2c2:	bf 00 00 00 00       	mov    $0x0,%edi
     2c7:	be 00 00 00 00       	mov    $0x0,%esi
     2cc:	ba cc 02 00 00       	mov    $0x2cc,%edx
     2d1:	b9 00 00 00 00       	mov    $0x0,%ecx
     2d6:	e8 00 00 00 00       	callq  2db <MedianPartition+0x2db>
     2db:	bf 00 00 00 00       	mov    $0x0,%edi
     2e0:	be 00 00 00 00       	mov    $0x0,%esi
     2e5:	ba cd 02 00 00       	mov    $0x2cd,%edx
     2ea:	b9 00 00 00 00       	mov    $0x0,%ecx
     2ef:	e8 00 00 00 00       	callq  2f4 <MedianPartition+0x2f4>
     2f4:	bf 00 00 00 00       	mov    $0x0,%edi
     2f9:	be 00 00 00 00       	mov    $0x0,%esi
     2fe:	ba ce 02 00 00       	mov    $0x2ce,%edx
     303:	b9 00 00 00 00       	mov    $0x0,%ecx
     308:	e8 00 00 00 00       	callq  30d <MedianPartition+0x30d>
     30d:	bf 00 00 00 00       	mov    $0x0,%edi
     312:	be 00 00 00 00       	mov    $0x0,%esi
     317:	ba dc 02 00 00       	mov    $0x2dc,%edx
     31c:	b9 00 00 00 00       	mov    $0x0,%ecx
     321:	e8 00 00 00 00       	callq  326 <MedianPartition+0x326>
     326:	bf 00 00 00 00       	mov    $0x0,%edi
     32b:	be 00 00 00 00       	mov    $0x0,%esi
     330:	ba eb 02 00 00       	mov    $0x2eb,%edx
     335:	b9 00 00 00 00       	mov    $0x0,%ecx
     33a:	e8 00 00 00 00       	callq  33f <MedianPartition+0x33f>
     33f:	bf 00 00 00 00       	mov    $0x0,%edi
     344:	be 00 00 00 00       	mov    $0x0,%esi
     349:	ba f4 02 00 00       	mov    $0x2f4,%edx
     34e:	b9 00 00 00 00       	mov    $0x0,%ecx
     353:	e8 00 00 00 00       	callq  358 <MedianPartition+0x358>
     358:	bf 00 00 00 00       	mov    $0x0,%edi
     35d:	be 00 00 00 00       	mov    $0x0,%esi
     362:	ba 1c 03 00 00       	mov    $0x31c,%edx
     367:	b9 00 00 00 00       	mov    $0x0,%ecx
     36c:	e8 00 00 00 00       	callq  371 <MedianPartition+0x371>
     371:	bf 00 00 00 00       	mov    $0x0,%edi
     376:	be 00 00 00 00       	mov    $0x0,%esi
     37b:	ba 30 03 00 00       	mov    $0x330,%edx
     380:	b9 00 00 00 00       	mov    $0x0,%ecx
     385:	e8 00 00 00 00       	callq  38a <MedianPartition+0x38a>
     38a:	bf 00 00 00 00       	mov    $0x0,%edi
     38f:	be 00 00 00 00       	mov    $0x0,%esi
     394:	ba 37 03 00 00       	mov    $0x337,%edx
     399:	b9 00 00 00 00       	mov    $0x0,%ecx
     39e:	e8 00 00 00 00       	callq  3a3 <MedianPartition+0x3a3>
     3a3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
     3aa:	84 00 00 00 00 00 

00000000000003b0 <Partition>:
     3b0:	55                   	push   %rbp
     3b1:	48 89 e5             	mov    %rsp,%rbp
     3b4:	41 57                	push   %r15
     3b6:	41 56                	push   %r14
     3b8:	41 55                	push   %r13
     3ba:	41 54                	push   %r12
     3bc:	53                   	push   %rbx
     3bd:	48 81 e4 e0 ff ff ff 	and    $0xffffffffffffffe0,%rsp
     3c4:	48 81 ec 20 01 00 00 	sub    $0x120,%rsp
     3cb:	48 89 bc 24 98 00 00 	mov    %rdi,0x98(%rsp)
     3d2:	00 
     3d3:	48 89 f0             	mov    %rsi,%rax
     3d6:	48 83 e0 1f          	and    $0x1f,%rax
     3da:	bf 20 00 00 00       	mov    $0x20,%edi
     3df:	48 29 c7             	sub    %rax,%rdi
     3e2:	45 31 e4             	xor    %r12d,%r12d
     3e5:	48 89 f0             	mov    %rsi,%rax
     3e8:	48 83 e0 1f          	and    $0x1f,%rax
     3ec:	4c 0f 45 e7          	cmovne %rdi,%r12
     3f0:	4d 39 c4             	cmp    %r8,%r12
     3f3:	4d 89 c2             	mov    %r8,%r10
     3f6:	4d 0f 42 d4          	cmovb  %r12,%r10
     3fa:	44 89 c0             	mov    %r8d,%eax
     3fd:	44 29 d0             	sub    %r10d,%eax
     400:	48 83 e0 1f          	and    $0x1f,%rax
     404:	4c 01 d0             	add    %r10,%rax
     407:	4c 39 c0             	cmp    %r8,%rax
     40a:	0f 83 e6 02 00 00    	jae    6f6 <Partition+0x346>
     410:	48 89 54 24 78       	mov    %rdx,0x78(%rsp)
     415:	48 89 8c 24 80 00 00 	mov    %rcx,0x80(%rsp)
     41c:	00 
     41d:	4d 89 c7             	mov    %r8,%r15
     420:	49 29 c7             	sub    %rax,%r15
     423:	4c 89 f9             	mov    %r15,%rcx
     426:	48 c1 e9 05          	shr    $0x5,%rcx
     42a:	0f 31                	rdtsc  
     42c:	89 c0                	mov    %eax,%eax
     42e:	48 85 c9             	test   %rcx,%rcx
     431:	0f 84 42 0b 00 00    	je     f79 <Partition+0xbc9>
     437:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
     43c:	48 89 54 24 50       	mov    %rdx,0x50(%rsp)
     441:	48 89 8c 24 88 00 00 	mov    %rcx,0x88(%rsp)
     448:	00 
     449:	4a 8d 3c 16          	lea    (%rsi,%r10,1),%rdi
     44d:	48 89 7c 24 60       	mov    %rdi,0x60(%rsp)
     452:	0f 31                	rdtsc  
     454:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
     459:	41 0f b6 c9          	movzbl %r9b,%ecx
     45d:	44 89 4c 24 1c       	mov    %r9d,0x1c(%rsp)
     462:	89 ca                	mov    %ecx,%edx
     464:	c1 e2 08             	shl    $0x8,%edx
     467:	09 ca                	or     %ecx,%edx
     469:	89 cb                	mov    %ecx,%ebx
     46b:	c1 e3 10             	shl    $0x10,%ebx
     46e:	09 d3                	or     %edx,%ebx
     470:	c1 e1 18             	shl    $0x18,%ecx
     473:	09 d9                	or     %ebx,%ecx
     475:	89 8c 24 04 01 00 00 	mov    %ecx,0x104(%rsp)
     47c:	c4 e2 7d 18 8c 24 04 	vbroadcastss 0x104(%rsp),%ymm1
     483:	01 00 00 
     486:	c5 7d 6f 0d 00 00 00 	vmovdqa 0x0(%rip),%ymm9        # 48e <Partition+0xde>
     48d:	00 
     48e:	c4 c1 75 fc d1       	vpaddb %ymm9,%ymm1,%ymm2
     493:	c5 fd 6f 0f          	vmovdqa (%rdi),%ymm1
     497:	c4 c1 75 fc d9       	vpaddb %ymm9,%ymm1,%ymm3
     49c:	c5 e5 64 da          	vpcmpgtb %ymm2,%ymm3,%ymm3
     4a0:	c5 fd d7 d3          	vpmovmskb %ymm3,%edx
     4a4:	44 0f b6 f2          	movzbl %dl,%r14d
     4a8:	89 d7                	mov    %edx,%edi
     4aa:	44 0f b7 da          	movzwl %dx,%r11d
     4ae:	f3 44 0f b8 ca       	popcnt %edx,%r9d
     4b3:	48 8b 9c 24 98 00 00 	mov    0x98(%rsp),%rbx
     4ba:	00 
     4bb:	4c 8b 2b             	mov    (%rbx),%r13
     4be:	45 89 4d 00          	mov    %r9d,0x0(%r13)
     4c2:	0f b6 de             	movzbl %dh,%ebx
     4c5:	c1 ea 18             	shr    $0x18,%edx
     4c8:	c5 fa 7e 1c d5 00 00 	vmovq  0x0(,%rdx,8),%xmm3
     4cf:	00 00 
     4d1:	c5 fa 7e 24 dd 00 00 	vmovq  0x0(,%rbx,8),%xmm4
     4d8:	00 00 
     4da:	c4 a1 7a 7e 2c f5 00 	vmovq  0x0(,%r14,8),%xmm5
     4e1:	00 00 00 
     4e4:	f3 41 0f b8 d6       	popcnt %r14d,%edx
     4e9:	c1 ef 10             	shr    $0x10,%edi
     4ec:	40 0f b6 ff          	movzbl %dil,%edi
     4f0:	c5 fa 7e 34 fd 00 00 	vmovq  0x0(,%rdi,8),%xmm6
     4f7:	00 00 
     4f9:	f3 44 0f b8 f7       	popcnt %edi,%r14d
     4fe:	f3 45 0f b8 db       	popcnt %r11d,%r11d
     503:	49 83 ff 40          	cmp    $0x40,%r15
     507:	c5 d1 6c e4          	vpunpcklqdq %xmm4,%xmm5,%xmm4
     50b:	89 c0                	mov    %eax,%eax
     50d:	c5 c9 6c db          	vpunpcklqdq %xmm3,%xmm6,%xmm3
     511:	c4 e3 5d 38 db 01    	vinserti128 $0x1,%xmm3,%ymm4,%ymm3
     517:	c5 e5 fc 1d 00 00 00 	vpaddb 0x0(%rip),%ymm3,%ymm3        # 51f <Partition+0x16f>
     51e:	00 
     51f:	0f 82 47 02 00 00    	jb     76c <Partition+0x3bc>
     525:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
     52c:	00 
     52d:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
     531:	4c 89 54 24 70       	mov    %r10,0x70(%rsp)
     536:	4c 89 e0             	mov    %r12,%rax
     539:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
     53e:	48 f7 d0             	not    %rax
     541:	4d 89 c7             	mov    %r8,%r15
     544:	49 f7 d7             	not    %r15
     547:	4c 39 f8             	cmp    %r15,%rax
     54a:	4c 0f 47 f8          	cmova  %rax,%r15
     54e:	4f 8d 4c 38 01       	lea    0x1(%r8,%r15,1),%r9
     553:	4c 89 c8             	mov    %r9,%rax
     556:	48 83 e0 1f          	and    $0x1f,%rax
     55a:	49 29 c1             	sub    %rax,%r9
     55d:	49 c1 e9 05          	shr    $0x5,%r9
     561:	49 83 c1 fe          	add    $0xfffffffffffffffe,%r9
     565:	41 bc 1f 00 00 00    	mov    $0x1f,%r12d
     56b:	4d 29 fc             	sub    %r15,%r12
     56e:	49 01 f4             	add    %rsi,%r12
     571:	48 89 74 24 68       	mov    %rsi,0x68(%rsp)
     576:	49 f7 d7             	not    %r15
     579:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
     580:	00 
     581:	4a 8d 34 38          	lea    (%rax,%r15,1),%rsi
     585:	4c 03 7c 24 78       	add    0x78(%rsp),%r15
     58a:	31 c0                	xor    %eax,%eax
     58c:	c5 7d 6f 15 00 00 00 	vmovdqa 0x0(%rip),%ymm10        # 594 <Partition+0x1e4>
     593:	00 
     594:	eb 1c                	jmp    5b2 <Partition+0x202>
     596:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
     59d:	00 00 00 
     5a0:	48 8b 8c 24 98 00 00 	mov    0x98(%rsp),%rcx
     5a7:	00 
     5a8:	4c 8b 29             	mov    (%rcx),%r13
     5ab:	49 ff c9             	dec    %r9
     5ae:	48 83 c0 04          	add    $0x4,%rax
     5b2:	c5 fd 6f e9          	vmovdqa %ymm1,%ymm5
     5b6:	c4 c1 7d 6f 0c c4    	vmovdqa (%r12,%rax,8),%ymm1
     5bc:	c4 c1 75 fc f1       	vpaddb %ymm9,%ymm1,%ymm6
     5c1:	c5 cd 64 f2          	vpcmpgtb %ymm2,%ymm6,%ymm6
     5c5:	c5 fd d7 de          	vpmovmskb %ymm6,%ebx
     5c9:	44 0f b6 d3          	movzbl %bl,%r10d
     5cd:	f3 0f b8 cb          	popcnt %ebx,%ecx
     5d1:	41 89 4c 05 04       	mov    %ecx,0x4(%r13,%rax,1)
     5d6:	89 d9                	mov    %ebx,%ecx
     5d8:	0f b6 ff             	movzbl %bh,%edi
     5db:	c5 fa 7e 34 fd 00 00 	vmovq  0x0(,%rdi,8),%xmm6
     5e2:	00 00 
     5e4:	0f b7 fb             	movzwl %bx,%edi
     5e7:	c1 eb 18             	shr    $0x18,%ebx
     5ea:	c5 fa 7e 3c dd 00 00 	vmovq  0x0(,%rbx,8),%xmm7
     5f1:	00 00 
     5f3:	ff c2                	inc    %edx
     5f5:	48 c1 e2 04          	shl    $0x4,%rdx
     5f9:	c5 7e 6f 82 00 00 00 	vmovdqu 0x0(%rdx),%ymm8
     600:	00 
     601:	44 89 f2             	mov    %r14d,%edx
     604:	44 89 db             	mov    %r11d,%ebx
     607:	89 d2                	mov    %edx,%edx
     609:	48 c1 e2 04          	shl    $0x4,%rdx
     60d:	c4 63 3d 02 82 00 00 	vpblendd $0xf0,0x0(%rdx),%ymm8,%ymm8
     614:	00 00 f0 
     617:	89 da                	mov    %ebx,%edx
     619:	83 e2 0f             	and    $0xf,%edx
     61c:	48 c1 e2 05          	shl    $0x5,%rdx
     620:	c4 c2 65 00 d8       	vpshufb %ymm8,%ymm3,%ymm3
     625:	c4 e2 65 00 9a 00 00 	vpshufb 0x0(%rdx),%ymm3,%ymm3
     62c:	00 00 
     62e:	89 da                	mov    %ebx,%edx
     630:	48 c1 e2 05          	shl    $0x5,%rdx
     634:	c5 7d 6f 82 00 00 00 	vmovdqa 0x0(%rdx),%ymm8
     63b:	00 
     63c:	c4 a1 7a 7e 04 d5 00 	vmovq  0x0(,%r10,8),%xmm0
     643:	00 00 00 
     646:	f3 41 0f b8 d2       	popcnt %r10d,%edx
     64b:	c1 e9 10             	shr    $0x10,%ecx
     64e:	0f b6 c9             	movzbl %cl,%ecx
     651:	c5 fa 7e 24 cd 00 00 	vmovq  0x0(,%rcx,8),%xmm4
     658:	00 00 
     65a:	f3 44 0f b8 f1       	popcnt %ecx,%r14d
     65f:	f3 44 0f b8 df       	popcnt %edi,%r11d
     664:	c5 f9 6c c6          	vpunpcklqdq %xmm6,%xmm0,%xmm0
     668:	c5 d9 6c e7          	vpunpcklqdq %xmm7,%xmm4,%xmm4
     66c:	c4 e3 7d 38 c4 01    	vinserti128 $0x1,%xmm4,%ymm0,%ymm0
     672:	c4 e2 55 00 e3       	vpshufb %ymm3,%ymm5,%ymm4
     677:	c4 c1 7d 6f 2c c7    	vmovdqa (%r15,%rax,8),%ymm5
     67d:	c4 e2 55 00 eb       	vpshufb %ymm3,%ymm5,%ymm5
     682:	c5 fd 6f 34 c6       	vmovdqa (%rsi,%rax,8),%ymm6
     687:	c4 e2 4d 00 db       	vpshufb %ymm3,%ymm6,%ymm3
     68c:	c4 e3 5d 46 f4 01    	vperm2i128 $0x1,%ymm4,%ymm4,%ymm6
     692:	c4 e3 55 46 fd 01    	vperm2i128 $0x1,%ymm5,%ymm5,%ymm7
     698:	c4 63 65 46 db 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm11
     69e:	c5 bd db e4          	vpand  %ymm4,%ymm8,%ymm4
     6a2:	c5 bd db ed          	vpand  %ymm5,%ymm8,%ymm5
     6a6:	c5 bd db db          	vpand  %ymm3,%ymm8,%ymm3
     6aa:	c5 bd df f6          	vpandn %ymm6,%ymm8,%ymm6
     6ae:	c5 bd df ff          	vpandn %ymm7,%ymm8,%ymm7
     6b2:	c4 41 3d df c3       	vpandn %ymm11,%ymm8,%ymm8
     6b7:	c5 dd eb e6          	vpor   %ymm6,%ymm4,%ymm4
     6bb:	c5 d5 eb ef          	vpor   %ymm7,%ymm5,%ymm5
     6bf:	c4 c1 65 eb d8       	vpor   %ymm8,%ymm3,%ymm3
     6c4:	c4 c1 7d 7f 64 c4 e0 	vmovdqa %ymm4,-0x20(%r12,%rax,8)
     6cb:	c4 c1 7d 7f 2c c7    	vmovdqa %ymm5,(%r15,%rax,8)
     6d1:	c5 fd 7f 1c c6       	vmovdqa %ymm3,(%rsi,%rax,8)
     6d6:	c4 c1 7d fc da       	vpaddb %ymm10,%ymm0,%ymm3
     6db:	4d 85 c9             	test   %r9,%r9
     6de:	0f 85 bc fe ff ff    	jne    5a0 <Partition+0x1f0>
     6e4:	4c 8b 54 24 70       	mov    0x70(%rsp),%r10
     6e9:	48 8b 84 24 90 00 00 	mov    0x90(%rsp),%rax
     6f0:	00 
     6f1:	e9 84 00 00 00       	jmpq   77a <Partition+0x3ca>
     6f6:	31 c0                	xor    %eax,%eax
     6f8:	4d 85 c0             	test   %r8,%r8
     6fb:	0f 84 66 08 00 00    	je     f67 <Partition+0xbb7>
     701:	45 31 d2             	xor    %r10d,%r10d
     704:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
     70b:	00 00 00 00 00 
     710:	42 8a 3c 16          	mov    (%rsi,%r10,1),%dil
     714:	44 38 cf             	cmp    %r9b,%dil
     717:	76 43                	jbe    75c <Partition+0x3ac>
     719:	49 8d 40 ff          	lea    -0x1(%r8),%rax
     71d:	42 8a 5c 06 ff       	mov    -0x1(%rsi,%r8,1),%bl
     722:	42 88 1c 16          	mov    %bl,(%rsi,%r10,1)
     726:	42 88 7c 06 ff       	mov    %dil,-0x1(%rsi,%r8,1)
     72b:	42 8a 3c 12          	mov    (%rdx,%r10,1),%dil
     72f:	42 8a 5c 02 ff       	mov    -0x1(%rdx,%r8,1),%bl
     734:	42 88 1c 12          	mov    %bl,(%rdx,%r10,1)
     738:	42 88 7c 02 ff       	mov    %dil,-0x1(%rdx,%r8,1)
     73d:	42 8a 3c 11          	mov    (%rcx,%r10,1),%dil
     741:	42 8a 5c 01 ff       	mov    -0x1(%rcx,%r8,1),%bl
     746:	42 88 1c 11          	mov    %bl,(%rcx,%r10,1)
     74a:	42 88 7c 01 ff       	mov    %dil,-0x1(%rcx,%r8,1)
     74f:	49 89 c0             	mov    %rax,%r8
     752:	49 39 c2             	cmp    %rax,%r10
     755:	72 b9                	jb     710 <Partition+0x360>
     757:	e9 0b 08 00 00       	jmpq   f67 <Partition+0xbb7>
     75c:	49 ff c2             	inc    %r10
     75f:	4c 89 c0             	mov    %r8,%rax
     762:	4d 39 c2             	cmp    %r8,%r10
     765:	72 a9                	jb     710 <Partition+0x360>
     767:	e9 fb 07 00 00       	jmpq   f67 <Partition+0xbb7>
     76c:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
     771:	89 4c 24 3c          	mov    %ecx,0x3c(%rsp)
     775:	48 89 74 24 68       	mov    %rsi,0x68(%rsp)
     77a:	4c 89 54 24 70       	mov    %r10,0x70(%rsp)
     77f:	4c 8b 4d 10          	mov    0x10(%rbp),%r9
     783:	4c 8b 7c 24 78       	mov    0x78(%rsp),%r15
     788:	4f 8d 24 17          	lea    (%r15,%r10,1),%r12
     78c:	4c 8b ac 24 80 00 00 	mov    0x80(%rsp),%r13
     793:	00 
     794:	4b 8d 74 15 00       	lea    0x0(%r13,%r10,1),%rsi
     799:	48 89 b4 24 90 00 00 	mov    %rsi,0x90(%rsp)
     7a0:	00 
     7a1:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
     7a6:	48 c1 e1 20          	shl    $0x20,%rcx
     7aa:	48 09 c1             	or     %rax,%rcx
     7ad:	44 89 d8             	mov    %r11d,%eax
     7b0:	83 e0 0f             	and    $0xf,%eax
     7b3:	48 c1 e0 05          	shl    $0x5,%rax
     7b7:	ff c2                	inc    %edx
     7b9:	48 c1 e2 04          	shl    $0x4,%rdx
     7bd:	c5 fe 6f 82 00 00 00 	vmovdqu 0x0(%rdx),%ymm0
     7c4:	00 
     7c5:	49 c1 e6 04          	shl    $0x4,%r14
     7c9:	c4 c3 7d 02 86 00 00 	vpblendd $0xf0,0x0(%r14),%ymm0,%ymm0
     7d0:	00 00 f0 
     7d3:	c4 e2 65 00 c0       	vpshufb %ymm0,%ymm3,%ymm0
     7d8:	c4 e2 7d 00 80 00 00 	vpshufb 0x0(%rax),%ymm0,%ymm0
     7df:	00 00 
     7e1:	48 8b bc 24 88 00 00 	mov    0x88(%rsp),%rdi
     7e8:	00 
     7e9:	4c 8d 77 ff          	lea    -0x1(%rdi),%r14
     7ed:	4c 89 f0             	mov    %r14,%rax
     7f0:	48 c1 e0 05          	shl    $0x5,%rax
     7f4:	c4 e2 75 00 c8       	vpshufb %ymm0,%ymm1,%ymm1
     7f9:	c4 c1 7d 6f 14 04    	vmovdqa (%r12,%rax,1),%ymm2
     7ff:	c4 e2 6d 00 d0       	vpshufb %ymm0,%ymm2,%ymm2
     804:	c5 fd 6f 1c 06       	vmovdqa (%rsi,%rax,1),%ymm3
     809:	c4 e2 65 00 c0       	vpshufb %ymm0,%ymm3,%ymm0
     80e:	49 c1 e3 05          	shl    $0x5,%r11
     812:	c4 c1 7d 6f 9b 00 00 	vmovdqa 0x0(%r11),%ymm3
     819:	00 00 
     81b:	c4 e3 75 46 e1 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm4
     821:	c4 e3 6d 46 ea 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm5
     827:	c4 e3 7d 46 f0 01    	vperm2i128 $0x1,%ymm0,%ymm0,%ymm6
     82d:	c5 e5 db c9          	vpand  %ymm1,%ymm3,%ymm1
     831:	c5 e5 db d2          	vpand  %ymm2,%ymm3,%ymm2
     835:	c5 e5 db c0          	vpand  %ymm0,%ymm3,%ymm0
     839:	c5 e5 df e4          	vpandn %ymm4,%ymm3,%ymm4
     83d:	c5 e5 df ed          	vpandn %ymm5,%ymm3,%ymm5
     841:	c5 e5 df de          	vpandn %ymm6,%ymm3,%ymm3
     845:	c5 f5 eb cc          	vpor   %ymm4,%ymm1,%ymm1
     849:	c5 ed eb d5          	vpor   %ymm5,%ymm2,%ymm2
     84d:	c5 fd eb c3          	vpor   %ymm3,%ymm0,%ymm0
     851:	48 8b 5c 24 60       	mov    0x60(%rsp),%rbx
     856:	c5 fd 7f 0c 03       	vmovdqa %ymm1,(%rbx,%rax,1)
     85b:	c4 c1 7d 7f 14 04    	vmovdqa %ymm2,(%r12,%rax,1)
     861:	c5 fd 7f 04 06       	vmovdqa %ymm0,(%rsi,%rax,1)
     866:	0f 31                	rdtsc  
     868:	89 c0                	mov    %eax,%eax
     86a:	48 c1 e2 20          	shl    $0x20,%rdx
     86e:	48 09 c2             	or     %rax,%rdx
     871:	49 8b 41 50          	mov    0x50(%r9),%rax
     875:	48 29 c8             	sub    %rcx,%rax
     878:	48 01 d0             	add    %rdx,%rax
     87b:	49 89 41 50          	mov    %rax,0x50(%r9)
     87f:	48 c1 e7 05          	shl    $0x5,%rdi
     883:	48 89 bc 24 88 00 00 	mov    %rdi,0x88(%rsp)
     88a:	00 
     88b:	49 01 79 58          	add    %rdi,0x58(%r9)
     88f:	0f 31                	rdtsc  
     891:	48 89 54 24 58       	mov    %rdx,0x58(%rsp)
     896:	89 c0                	mov    %eax,%eax
     898:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
     89d:	c5 fd 6f 33          	vmovdqa (%rbx),%ymm6
     8a1:	49 89 da             	mov    %rbx,%r10
     8a4:	c4 c1 7d 6f 2c 24    	vmovdqa (%r12),%ymm5
     8aa:	c5 fd 6f 26          	vmovdqa (%rsi),%ymm4
     8ae:	4d 85 f6             	test   %r14,%r14
     8b1:	0f 84 f3 01 00 00    	je     aaa <Partition+0x6fa>
     8b7:	48 8b 84 24 98 00 00 	mov    0x98(%rsp),%rax
     8be:	00 
     8bf:	48 8b 00             	mov    (%rax),%rax
     8c2:	8b 38                	mov    (%rax),%edi
     8c4:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
     8c9:	48 f7 d0             	not    %rax
     8cc:	4c 89 c1             	mov    %r8,%rcx
     8cf:	48 f7 d1             	not    %rcx
     8d2:	48 39 c8             	cmp    %rcx,%rax
     8d5:	48 0f 47 c8          	cmova  %rax,%rcx
     8d9:	48 f7 d1             	not    %rcx
     8dc:	49 8d 44 0d 00       	lea    0x0(%r13,%rcx,1),%rax
     8e1:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
     8e6:	49 8d 04 0f          	lea    (%r15,%rcx,1),%rax
     8ea:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
     8ef:	48 03 4c 24 68       	add    0x68(%rsp),%rcx
     8f4:	48 89 4c 24 30       	mov    %rcx,0x30(%rsp)
     8f9:	45 31 db             	xor    %r11d,%r11d
     8fc:	4c 89 d8             	mov    %r11,%rax
     8ff:	48 c1 e0 05          	shl    $0x5,%rax
     903:	4c 8b 4c 24 30       	mov    0x30(%rsp),%r9
     908:	4c 8b 6c 24 20       	mov    0x20(%rsp),%r13
     90d:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
     912:	c5 fd 6f ce          	vmovdqa %ymm6,%ymm1
     916:	c5 fd 6f d5          	vmovdqa %ymm5,%ymm2
     91a:	c5 fd 6f dc          	vmovdqa %ymm4,%ymm3
     91e:	41 89 ff             	mov    %edi,%r15d
     921:	4c 89 f7             	mov    %r14,%rdi
     924:	66 66 66 2e 0f 1f 84 	data16 data16 nopw %cs:0x0(%rax,%rax,1)
     92b:	00 00 00 00 00 
     930:	48 8b 8c 24 98 00 00 	mov    0x98(%rsp),%rcx
     937:	00 
     938:	48 8b 09             	mov    (%rcx),%rcx
     93b:	8b 1c b9             	mov    (%rcx,%rdi,4),%ebx
     93e:	48 c1 e7 05          	shl    $0x5,%rdi
     942:	44 89 f9             	mov    %r15d,%ecx
     945:	83 e1 0f             	and    $0xf,%ecx
     948:	48 c1 e1 05          	shl    $0x5,%rcx
     94c:	44 89 fe             	mov    %r15d,%esi
     94f:	48 c1 e6 05          	shl    $0x5,%rsi
     953:	c5 fd 6f 86 00 00 00 	vmovdqa 0x0(%rsi),%ymm0
     95a:	00 
     95b:	c4 c1 7d 6f 24 3a    	vmovdqa (%r10,%rdi,1),%ymm4
     961:	c5 fd 6f a9 00 00 00 	vmovdqa 0x0(%rcx),%ymm5
     968:	00 
     969:	c4 e2 5d 00 e5       	vpshufb %ymm5,%ymm4,%ymm4
     96e:	c4 c1 7d 6f 34 3c    	vmovdqa (%r12,%rdi,1),%ymm6
     974:	c4 e2 4d 00 f5       	vpshufb %ymm5,%ymm6,%ymm6
     979:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
     980:	00 
     981:	c5 fd 6f 3c 39       	vmovdqa (%rcx,%rdi,1),%ymm7
     986:	c4 e2 45 00 ed       	vpshufb %ymm5,%ymm7,%ymm5
     98b:	c4 e3 5d 46 fc 01    	vperm2i128 $0x1,%ymm4,%ymm4,%ymm7
     991:	c4 63 4d 46 c6 01    	vperm2i128 $0x1,%ymm6,%ymm6,%ymm8
     997:	c4 63 55 46 d5 01    	vperm2i128 $0x1,%ymm5,%ymm5,%ymm10
     99d:	c5 dd db e0          	vpand  %ymm0,%ymm4,%ymm4
     9a1:	c5 cd db f0          	vpand  %ymm0,%ymm6,%ymm6
     9a5:	c5 d5 db e8          	vpand  %ymm0,%ymm5,%ymm5
     9a9:	c5 fd df ff          	vpandn %ymm7,%ymm0,%ymm7
     9ad:	c4 41 7d df c0       	vpandn %ymm8,%ymm0,%ymm8
     9b2:	c4 c1 7d df c2       	vpandn %ymm10,%ymm0,%ymm0
     9b7:	c5 dd eb ff          	vpor   %ymm7,%ymm4,%ymm7
     9bb:	c4 41 4d eb c0       	vpor   %ymm8,%ymm6,%ymm8
     9c0:	c5 d5 eb c0          	vpor   %ymm0,%ymm5,%ymm0
     9c4:	c5 7d 6f 96 00 00 00 	vmovdqa 0x0(%rsi),%ymm10
     9cb:	00 
     9cc:	c5 ad df e7          	vpandn %ymm7,%ymm10,%ymm4
     9d0:	c5 ad db e9          	vpand  %ymm1,%ymm10,%ymm5
     9d4:	c4 41 2d df d8       	vpandn %ymm8,%ymm10,%ymm11
     9d9:	c5 2d db e2          	vpand  %ymm2,%ymm10,%ymm12
     9dd:	c5 2d df e8          	vpandn %ymm0,%ymm10,%ymm13
     9e1:	c5 2d db f3          	vpand  %ymm3,%ymm10,%ymm14
     9e5:	c5 dd eb f5          	vpor   %ymm5,%ymm4,%ymm6
     9e9:	c4 c1 25 eb ec       	vpor   %ymm12,%ymm11,%ymm5
     9ee:	c4 c1 15 eb e6       	vpor   %ymm14,%ymm13,%ymm4
     9f3:	c5 ad db ff          	vpand  %ymm7,%ymm10,%ymm7
     9f7:	c5 ad df c9          	vpandn %ymm1,%ymm10,%ymm1
     9fb:	c4 41 2d db c0       	vpand  %ymm8,%ymm10,%ymm8
     a00:	c5 ad df d2          	vpandn %ymm2,%ymm10,%ymm2
     a04:	c5 ad db c0          	vpand  %ymm0,%ymm10,%ymm0
     a08:	c5 ad df db          	vpandn %ymm3,%ymm10,%ymm3
     a0c:	c5 f5 eb cf          	vpor   %ymm7,%ymm1,%ymm1
     a10:	c4 c1 6d eb d0       	vpor   %ymm8,%ymm2,%ymm2
     a15:	c5 e5 eb d8          	vpor   %ymm0,%ymm3,%ymm3
     a19:	42 8d 7c 3b e0       	lea    -0x20(%rbx,%r15,1),%edi
     a1e:	85 ff                	test   %edi,%edi
     a20:	b9 00 00 00 00       	mov    $0x0,%ecx
     a25:	0f 48 f9             	cmovs  %ecx,%edi
     a28:	85 ff                	test   %edi,%edi
     a2a:	75 44                	jne    a70 <Partition+0x6c0>
     a2c:	44 01 fb             	add    %r15d,%ebx
     a2f:	83 fb 20             	cmp    $0x20,%ebx
     a32:	b9 20 00 00 00       	mov    $0x20,%ecx
     a37:	0f 47 d9             	cmova  %ecx,%ebx
     a3a:	c4 c1 7d 7f 34 01    	vmovdqa %ymm6,(%r9,%rax,1)
     a40:	c4 c1 7d 7f 6c 05 00 	vmovdqa %ymm5,0x0(%r13,%rax,1)
     a47:	c5 fd 7f 24 02       	vmovdqa %ymm4,(%rdx,%rax,1)
     a4c:	49 ff c3             	inc    %r11
     a4f:	48 83 c2 20          	add    $0x20,%rdx
     a53:	49 83 c5 20          	add    $0x20,%r13
     a57:	49 83 c1 20          	add    $0x20,%r9
     a5b:	41 89 df             	mov    %ebx,%r15d
     a5e:	4c 89 df             	mov    %r11,%rdi
     a61:	4d 39 f3             	cmp    %r14,%r11
     a64:	0f 82 c6 fe ff ff    	jb     930 <Partition+0x580>
     a6a:	eb 4d                	jmp    ab9 <Partition+0x709>
     a6c:	0f 1f 40 00          	nopl   0x0(%rax)
     a70:	4c 89 f0             	mov    %r14,%rax
     a73:	48 c1 e0 05          	shl    $0x5,%rax
     a77:	c4 c1 7d 7f 0c 02    	vmovdqa %ymm1,(%r10,%rax,1)
     a7d:	c4 c1 7d 7f 14 04    	vmovdqa %ymm2,(%r12,%rax,1)
     a83:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
     a8a:	00 
     a8b:	c5 fd 7f 1c 01       	vmovdqa %ymm3,(%rcx,%rax,1)
     a90:	49 ff ce             	dec    %r14
     a93:	c5 fd 6f ce          	vmovdqa %ymm6,%ymm1
     a97:	c5 fd 6f d5          	vmovdqa %ymm5,%ymm2
     a9b:	c5 fd 6f dc          	vmovdqa %ymm4,%ymm3
     a9f:	4d 39 f3             	cmp    %r14,%r11
     aa2:	0f 82 54 fe ff ff    	jb     8fc <Partition+0x54c>
     aa8:	eb 0f                	jmp    ab9 <Partition+0x709>
     aaa:	c5 fd 6f ce          	vmovdqa %ymm6,%ymm1
     aae:	c5 fd 6f d5          	vmovdqa %ymm5,%ymm2
     ab2:	c5 fd 6f dc          	vmovdqa %ymm4,%ymm3
     ab6:	45 31 db             	xor    %r11d,%r11d
     ab9:	48 8b 74 24 50       	mov    0x50(%rsp),%rsi
     abe:	48 c1 e6 20          	shl    $0x20,%rsi
     ac2:	48 03 74 24 48       	add    0x48(%rsp),%rsi
     ac7:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
     acc:	48 c1 e1 20          	shl    $0x20,%rcx
     ad0:	48 03 4c 24 40       	add    0x40(%rsp),%rcx
     ad5:	49 c1 e3 05          	shl    $0x5,%r11
     ad9:	c4 81 7d 7f 0c 1a    	vmovdqa %ymm1,(%r10,%r11,1)
     adf:	c4 81 7d 7f 14 1c    	vmovdqa %ymm2,(%r12,%r11,1)
     ae5:	48 8b 84 24 90 00 00 	mov    0x90(%rsp),%rax
     aec:	00 
     aed:	c4 a1 7d 7f 1c 18    	vmovdqa %ymm3,(%rax,%r11,1)
     af3:	0f 31                	rdtsc  
     af5:	89 c0                	mov    %eax,%eax
     af7:	48 c1 e2 20          	shl    $0x20,%rdx
     afb:	48 09 c2             	or     %rax,%rdx
     afe:	48 8b 45 10          	mov    0x10(%rbp),%rax
     b02:	48 89 c7             	mov    %rax,%rdi
     b05:	48 8b 47 60          	mov    0x60(%rdi),%rax
     b09:	48 29 c8             	sub    %rcx,%rax
     b0c:	48 01 d0             	add    %rdx,%rax
     b0f:	48 89 47 60          	mov    %rax,0x60(%rdi)
     b13:	48 8b 8c 24 88 00 00 	mov    0x88(%rsp),%rcx
     b1a:	00 
     b1b:	48 01 4f 68          	add    %rcx,0x68(%rdi)
     b1f:	0f 31                	rdtsc  
     b21:	89 c0                	mov    %eax,%eax
     b23:	48 c1 e2 20          	shl    $0x20,%rdx
     b27:	48 09 c2             	or     %rax,%rdx
     b2a:	48 8b 47 40          	mov    0x40(%rdi),%rax
     b2e:	48 29 f0             	sub    %rsi,%rax
     b31:	48 01 d0             	add    %rdx,%rax
     b34:	48 89 47 40          	mov    %rax,0x40(%rdi)
     b38:	48 01 4f 48          	add    %rcx,0x48(%rdi)
     b3c:	4c 03 5c 24 70       	add    0x70(%rsp),%r11
     b41:	4d 39 c3             	cmp    %r8,%r11
     b44:	73 2a                	jae    b70 <Partition+0x7c0>
     b46:	4c 8b b4 24 80 00 00 	mov    0x80(%rsp),%r14
     b4d:	00 
     b4e:	4c 8b 7c 24 78       	mov    0x78(%rsp),%r15
     b53:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
     b58:	8b 4c 24 1c          	mov    0x1c(%rsp),%ecx
     b5c:	0f 1f 40 00          	nopl   0x0(%rax)
     b60:	42 38 0c 18          	cmp    %cl,(%rax,%r11,1)
     b64:	77 1c                	ja     b82 <Partition+0x7d2>
     b66:	49 ff c3             	inc    %r11
     b69:	4d 39 c3             	cmp    %r8,%r11
     b6c:	72 f2                	jb     b60 <Partition+0x7b0>
     b6e:	eb 12                	jmp    b82 <Partition+0x7d2>
     b70:	4c 8b b4 24 80 00 00 	mov    0x80(%rsp),%r14
     b77:	00 
     b78:	4c 8b 7c 24 78       	mov    0x78(%rsp),%r15
     b7d:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
     b82:	c5 f9 6e 44 24 3c    	vmovd  0x3c(%rsp),%xmm0
     b88:	45 31 e4             	xor    %r12d,%r12d
     b8b:	4c 89 d9             	mov    %r11,%rcx
     b8e:	48 83 c1 e0          	add    $0xffffffffffffffe0,%rcx
     b92:	4c 0f 49 e1          	cmovns %rcx,%r12
     b96:	c5 fe 6f 08          	vmovdqu (%rax),%ymm1
     b9a:	c5 fd 7f 8c 24 e0 00 	vmovdqa %ymm1,0xe0(%rsp)
     ba1:	00 00 
     ba3:	c4 c1 7e 6f 17       	vmovdqu (%r15),%ymm2
     ba8:	c5 fd 7f 94 24 c0 00 	vmovdqa %ymm2,0xc0(%rsp)
     baf:	00 00 
     bb1:	c4 c1 7e 6f 1e       	vmovdqu (%r14),%ymm3
     bb6:	c5 fd 7f 9c 24 a0 00 	vmovdqa %ymm3,0xa0(%rsp)
     bbd:	00 00 
     bbf:	c4 21 7e 6f 14 20    	vmovdqu (%rax,%r12,1),%ymm10
     bc5:	c4 01 7e 6f 1c 27    	vmovdqu (%r15,%r12,1),%ymm11
     bcb:	c4 81 7e 6f 34 26    	vmovdqu (%r14,%r12,1),%ymm6
     bd1:	c5 fa 11 84 24 04 01 	vmovss %xmm0,0x104(%rsp)
     bd8:	00 00 
     bda:	c4 e2 7d 18 bc 24 04 	vbroadcastss 0x104(%rsp),%ymm7
     be1:	01 00 00 
     be4:	c4 c1 45 fc f9       	vpaddb %ymm9,%ymm7,%ymm7
     be9:	c4 41 75 fc c1       	vpaddb %ymm9,%ymm1,%ymm8
     bee:	c5 bd 64 ff          	vpcmpgtb %ymm7,%ymm8,%ymm7
     bf2:	c5 fd d7 cf          	vpmovmskb %ymm7,%ecx
     bf6:	0f b6 d1             	movzbl %cl,%edx
     bf9:	f3 0f b8 f2          	popcnt %edx,%esi
     bfd:	89 cf                	mov    %ecx,%edi
     bff:	c1 ef 10             	shr    $0x10,%edi
     c02:	40 0f b6 ff          	movzbl %dil,%edi
     c06:	f3 44 0f b8 cf       	popcnt %edi,%r9d
     c0b:	0f b7 d9             	movzwl %cx,%ebx
     c0e:	f3 0f b8 db          	popcnt %ebx,%ebx
     c12:	49 89 c5             	mov    %rax,%r13
     c15:	f3 44 0f b8 d1       	popcnt %ecx,%r10d
     c1a:	c5 fa 7e 3c d5 00 00 	vmovq  0x0(,%rdx,8),%xmm7
     c21:	00 00 
     c23:	0f b6 d5             	movzbl %ch,%edx
     c26:	c5 fa 7e 24 d5 00 00 	vmovq  0x0(,%rdx,8),%xmm4
     c2d:	00 00 
     c2f:	c5 c1 6c e4          	vpunpcklqdq %xmm4,%xmm7,%xmm4
     c33:	c5 fa 7e 3c fd 00 00 	vmovq  0x0(,%rdi,8),%xmm7
     c3a:	00 00 
     c3c:	c1 e9 18             	shr    $0x18,%ecx
     c3f:	c5 fa 7e 2c cd 00 00 	vmovq  0x0(,%rcx,8),%xmm5
     c46:	00 00 
     c48:	c5 c1 6c ed          	vpunpcklqdq %xmm5,%xmm7,%xmm5
     c4c:	c4 e3 5d 38 e5 01    	vinserti128 $0x1,%xmm5,%ymm4,%ymm4
     c52:	c5 7d 6f 2d 00 00 00 	vmovdqa 0x0(%rip),%ymm13        # c5a <Partition+0x8aa>
     c59:	00 
     c5a:	c4 c1 5d fc e5       	vpaddb %ymm13,%ymm4,%ymm4
     c5f:	89 d9                	mov    %ebx,%ecx
     c61:	83 e1 0f             	and    $0xf,%ecx
     c64:	48 c1 e1 05          	shl    $0x5,%rcx
     c68:	ff c6                	inc    %esi
     c6a:	48 c1 e6 04          	shl    $0x4,%rsi
     c6e:	c5 fe 6f be 00 00 00 	vmovdqu 0x0(%rsi),%ymm7
     c75:	00 
     c76:	49 c1 e1 04          	shl    $0x4,%r9
     c7a:	c4 c3 45 02 b9 00 00 	vpblendd $0xf0,0x0(%r9),%ymm7,%ymm7
     c81:	00 00 f0 
     c84:	c4 e2 5d 00 e7       	vpshufb %ymm7,%ymm4,%ymm4
     c89:	c4 e2 5d 00 a1 00 00 	vpshufb 0x0(%rcx),%ymm4,%ymm4
     c90:	00 00 
     c92:	c4 e2 75 00 cc       	vpshufb %ymm4,%ymm1,%ymm1
     c97:	c4 e2 6d 00 d4       	vpshufb %ymm4,%ymm2,%ymm2
     c9c:	c4 e2 65 00 dc       	vpshufb %ymm4,%ymm3,%ymm3
     ca1:	48 c1 e3 05          	shl    $0x5,%rbx
     ca5:	c5 fd 6f a3 00 00 00 	vmovdqa 0x0(%rbx),%ymm4
     cac:	00 
     cad:	c4 e3 75 46 f9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm7
     cb3:	c4 63 6d 46 c2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm8
     cb9:	c4 63 65 46 e3 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm12
     cbf:	c5 dd db c9          	vpand  %ymm1,%ymm4,%ymm1
     cc3:	c5 dd db d2          	vpand  %ymm2,%ymm4,%ymm2
     cc7:	c5 dd db db          	vpand  %ymm3,%ymm4,%ymm3
     ccb:	c5 dd df ff          	vpandn %ymm7,%ymm4,%ymm7
     ccf:	c4 41 5d df c0       	vpandn %ymm8,%ymm4,%ymm8
     cd4:	c4 c1 5d df e4       	vpandn %ymm12,%ymm4,%ymm4
     cd9:	c5 f5 eb cf          	vpor   %ymm7,%ymm1,%ymm1
     cdd:	c4 c1 6d eb d0       	vpor   %ymm8,%ymm2,%ymm2
     ce2:	c5 e5 eb dc          	vpor   %ymm4,%ymm3,%ymm3
     ce6:	c5 fe 7f 8c 24 e0 00 	vmovdqu %ymm1,0xe0(%rsp)
     ced:	00 00 
     cef:	c5 fe 7f 94 24 c0 00 	vmovdqu %ymm2,0xc0(%rsp)
     cf6:	00 00 
     cf8:	c5 fe 7f 9c 24 a0 00 	vmovdqu %ymm3,0xa0(%rsp)
     cff:	00 00 
     d01:	b9 20 00 00 00       	mov    $0x20,%ecx
     d06:	4c 29 d1             	sub    %r10,%rcx
     d09:	4d 29 d3             	sub    %r10,%r11
     d0c:	4c 39 d9             	cmp    %r11,%rcx
     d0f:	4c 0f 4d d9          	cmovge %rcx,%r11
     d13:	c5 fd 6f 0d 00 00 00 	vmovdqa 0x0(%rip),%ymm1        # d1b <Partition+0x96b>
     d1a:	00 
     d1b:	c4 e2 2d 00 d1       	vpshufb %ymm1,%ymm10,%ymm2
     d20:	c4 e2 25 00 d9       	vpshufb %ymm1,%ymm11,%ymm3
     d25:	c4 e2 4d 00 e1       	vpshufb %ymm1,%ymm6,%ymm4
     d2a:	c4 e3 6d 46 d2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm2
     d30:	c4 e3 65 46 db 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm3
     d36:	c4 e3 5d 46 e4 01    	vperm2i128 $0x1,%ymm4,%ymm4,%ymm4
     d3c:	c4 c1 7e 7f 55 00    	vmovdqu %ymm2,0x0(%r13)
     d42:	c4 c1 7e 7f 1f       	vmovdqu %ymm3,(%r15)
     d47:	c4 c1 7e 7f 26       	vmovdqu %ymm4,(%r14)
     d4c:	c5 fc 28 94 24 e0 00 	vmovaps 0xe0(%rsp),%ymm2
     d53:	00 00 
     d55:	c4 81 7c 11 54 25 00 	vmovups %ymm2,0x0(%r13,%r12,1)
     d5c:	c5 fc 28 94 24 c0 00 	vmovaps 0xc0(%rsp),%ymm2
     d63:	00 00 
     d65:	c4 81 7c 11 14 27    	vmovups %ymm2,(%r15,%r12,1)
     d6b:	c5 fc 28 94 24 a0 00 	vmovaps 0xa0(%rsp),%ymm2
     d72:	00 00 
     d74:	c4 81 7c 11 14 26    	vmovups %ymm2,(%r14,%r12,1)
     d7a:	49 8d 48 e0          	lea    -0x20(%r8),%rcx
     d7e:	4c 39 d9             	cmp    %r11,%rcx
     d81:	49 0f 43 cb          	cmovae %r11,%rcx
     d85:	c4 81 7e 6f 54 05 e0 	vmovdqu -0x20(%r13,%r8,1),%ymm2
     d8c:	c5 fd 7f 94 24 e0 00 	vmovdqa %ymm2,0xe0(%rsp)
     d93:	00 00 
     d95:	c4 81 7e 6f 5c 07 e0 	vmovdqu -0x20(%r15,%r8,1),%ymm3
     d9c:	c5 fd 7f 9c 24 c0 00 	vmovdqa %ymm3,0xc0(%rsp)
     da3:	00 00 
     da5:	c4 81 7e 6f 64 06 e0 	vmovdqu -0x20(%r14,%r8,1),%ymm4
     dac:	c5 fd 7f a4 24 a0 00 	vmovdqa %ymm4,0xa0(%rsp)
     db3:	00 00 
     db5:	c4 41 7e 6f 54 0d 00 	vmovdqu 0x0(%r13,%rcx,1),%ymm10
     dbc:	c4 c1 7e 6f 3c 0f    	vmovdqu (%r15,%rcx,1),%ymm7
     dc2:	c4 41 7e 6f 04 0e    	vmovdqu (%r14,%rcx,1),%ymm8
     dc8:	c5 fa 11 84 24 04 01 	vmovss %xmm0,0x104(%rsp)
     dcf:	00 00 
     dd1:	c4 e2 7d 18 84 24 04 	vbroadcastss 0x104(%rsp),%ymm0
     dd8:	01 00 00 
     ddb:	c4 c1 7d fc c1       	vpaddb %ymm9,%ymm0,%ymm0
     de0:	c4 41 6d fc c9       	vpaddb %ymm9,%ymm2,%ymm9
     de5:	c5 b5 64 c0          	vpcmpgtb %ymm0,%ymm9,%ymm0
     de9:	c5 fd d7 c0          	vpmovmskb %ymm0,%eax
     ded:	0f b6 d0             	movzbl %al,%edx
     df0:	f3 0f b8 f2          	popcnt %edx,%esi
     df4:	89 c7                	mov    %eax,%edi
     df6:	c1 ef 10             	shr    $0x10,%edi
     df9:	40 0f b6 ff          	movzbl %dil,%edi
     dfd:	f3 44 0f b8 cf       	popcnt %edi,%r9d
     e02:	0f b7 d8             	movzwl %ax,%ebx
     e05:	f3 0f b8 db          	popcnt %ebx,%ebx
     e09:	f3 44 0f b8 d0       	popcnt %eax,%r10d
     e0e:	c5 fa 7e 04 d5 00 00 	vmovq  0x0(,%rdx,8),%xmm0
     e15:	00 00 
     e17:	0f b6 d4             	movzbl %ah,%edx
     e1a:	c5 fa 7e 2c d5 00 00 	vmovq  0x0(,%rdx,8),%xmm5
     e21:	00 00 
     e23:	c5 f9 6c c5          	vpunpcklqdq %xmm5,%xmm0,%xmm0
     e27:	c5 fa 7e 2c fd 00 00 	vmovq  0x0(,%rdi,8),%xmm5
     e2e:	00 00 
     e30:	c1 e8 18             	shr    $0x18,%eax
     e33:	c5 fa 7e 34 c5 00 00 	vmovq  0x0(,%rax,8),%xmm6
     e3a:	00 00 
     e3c:	c5 d1 6c ee          	vpunpcklqdq %xmm6,%xmm5,%xmm5
     e40:	c4 e3 7d 38 c5 01    	vinserti128 $0x1,%xmm5,%ymm0,%ymm0
     e46:	c4 c1 7d fc c5       	vpaddb %ymm13,%ymm0,%ymm0
     e4b:	89 d8                	mov    %ebx,%eax
     e4d:	83 e0 0f             	and    $0xf,%eax
     e50:	48 c1 e0 05          	shl    $0x5,%rax
     e54:	ff c6                	inc    %esi
     e56:	48 c1 e6 04          	shl    $0x4,%rsi
     e5a:	c5 fe 6f ae 00 00 00 	vmovdqu 0x0(%rsi),%ymm5
     e61:	00 
     e62:	49 c1 e1 04          	shl    $0x4,%r9
     e66:	c4 c3 55 02 a9 00 00 	vpblendd $0xf0,0x0(%r9),%ymm5,%ymm5
     e6d:	00 00 f0 
     e70:	c4 e2 7d 00 c5       	vpshufb %ymm5,%ymm0,%ymm0
     e75:	c4 e2 7d 00 80 00 00 	vpshufb 0x0(%rax),%ymm0,%ymm0
     e7c:	00 00 
     e7e:	c4 e2 6d 00 d0       	vpshufb %ymm0,%ymm2,%ymm2
     e83:	c4 e2 65 00 d8       	vpshufb %ymm0,%ymm3,%ymm3
     e88:	c4 e2 5d 00 c0       	vpshufb %ymm0,%ymm4,%ymm0
     e8d:	48 c1 e3 05          	shl    $0x5,%rbx
     e91:	c5 fd 6f a3 00 00 00 	vmovdqa 0x0(%rbx),%ymm4
     e98:	00 
     e99:	c4 e3 6d 46 ea 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm5
     e9f:	c4 e3 65 46 f3 01    	vperm2i128 $0x1,%ymm3,%ymm3,%ymm6
     ea5:	c4 63 7d 46 c8 01    	vperm2i128 $0x1,%ymm0,%ymm0,%ymm9
     eab:	c5 dd db d2          	vpand  %ymm2,%ymm4,%ymm2
     eaf:	c5 dd db db          	vpand  %ymm3,%ymm4,%ymm3
     eb3:	c5 dd db c0          	vpand  %ymm0,%ymm4,%ymm0
     eb7:	c5 dd df ed          	vpandn %ymm5,%ymm4,%ymm5
     ebb:	c5 dd df f6          	vpandn %ymm6,%ymm4,%ymm6
     ebf:	c4 c1 5d df e1       	vpandn %ymm9,%ymm4,%ymm4
     ec4:	c5 ed eb d5          	vpor   %ymm5,%ymm2,%ymm2
     ec8:	c5 e5 eb de          	vpor   %ymm6,%ymm3,%ymm3
     ecc:	c5 fd eb c4          	vpor   %ymm4,%ymm0,%ymm0
     ed0:	c5 fe 7f 94 24 e0 00 	vmovdqu %ymm2,0xe0(%rsp)
     ed7:	00 00 
     ed9:	c5 fe 7f 9c 24 c0 00 	vmovdqu %ymm3,0xc0(%rsp)
     ee0:	00 00 
     ee2:	c5 fe 7f 84 24 a0 00 	vmovdqu %ymm0,0xa0(%rsp)
     ee9:	00 00 
     eeb:	4c 89 c2             	mov    %r8,%rdx
     eee:	4c 29 d2             	sub    %r10,%rdx
     ef1:	b8 20 00 00 00       	mov    $0x20,%eax
     ef6:	44 29 d0             	sub    %r10d,%eax
     ef9:	4c 01 d8             	add    %r11,%rax
     efc:	48 39 c2             	cmp    %rax,%rdx
     eff:	48 0f 42 c2          	cmovb  %rdx,%rax
     f03:	c4 e2 2d 00 c1       	vpshufb %ymm1,%ymm10,%ymm0
     f08:	c4 e2 45 00 d1       	vpshufb %ymm1,%ymm7,%ymm2
     f0d:	c4 e2 3d 00 c9       	vpshufb %ymm1,%ymm8,%ymm1
     f12:	c4 e3 7d 46 c0 01    	vperm2i128 $0x1,%ymm0,%ymm0,%ymm0
     f18:	c4 e3 6d 46 d2 01    	vperm2i128 $0x1,%ymm2,%ymm2,%ymm2
     f1e:	c4 e3 75 46 c9 01    	vperm2i128 $0x1,%ymm1,%ymm1,%ymm1
     f24:	c4 81 7e 7f 44 05 e0 	vmovdqu %ymm0,-0x20(%r13,%r8,1)
     f2b:	c4 81 7e 7f 54 07 e0 	vmovdqu %ymm2,-0x20(%r15,%r8,1)
     f32:	c4 81 7e 7f 4c 06 e0 	vmovdqu %ymm1,-0x20(%r14,%r8,1)
     f39:	c5 fc 28 84 24 e0 00 	vmovaps 0xe0(%rsp),%ymm0
     f40:	00 00 
     f42:	c4 c1 7c 11 44 0d 00 	vmovups %ymm0,0x0(%r13,%rcx,1)
     f49:	c5 fc 28 84 24 c0 00 	vmovaps 0xc0(%rsp),%ymm0
     f50:	00 00 
     f52:	c4 c1 7c 11 04 0f    	vmovups %ymm0,(%r15,%rcx,1)
     f58:	c5 fc 28 84 24 a0 00 	vmovaps 0xa0(%rsp),%ymm0
     f5f:	00 00 
     f61:	c4 c1 7c 11 04 0e    	vmovups %ymm0,(%r14,%rcx,1)
     f67:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
     f6b:	5b                   	pop    %rbx
     f6c:	41 5c                	pop    %r12
     f6e:	41 5d                	pop    %r13
     f70:	41 5e                	pop    %r14
     f72:	41 5f                	pop    %r15
     f74:	5d                   	pop    %rbp
     f75:	c5 f8 77             	vzeroupper 
     f78:	c3                   	retq   
     f79:	bf 00 00 00 00       	mov    $0x0,%edi
     f7e:	be 00 00 00 00       	mov    $0x0,%esi
     f83:	ba 00 02 00 00       	mov    $0x200,%edx
     f88:	b9 00 00 00 00       	mov    $0x0,%ecx
     f8d:	e8 00 00 00 00       	callq  f92 <Partition+0xbe2>
     f92:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 nopw %cs:0x0(%rax,%rax,1)
     f99:	1f 84 00 00 00 00 00 

0000000000000fa0 <MPWorkspaceInit>:
     fa0:	53                   	push   %rbx
     fa1:	48 89 fb             	mov    %rdi,%rbx
     fa4:	48 85 db             	test   %rbx,%rbx
     fa7:	74 20                	je     fc9 <MPWorkspaceInit+0x29>
     fa9:	48 85 f6             	test   %rsi,%rsi
     fac:	74 34                	je     fe2 <MPWorkspaceInit+0x42>
     fae:	48 c1 ee 02          	shr    $0x2,%rsi
     fb2:	48 bf f8 ff ff ff ff 	movabs $0x3ffffffffffffff8,%rdi
     fb9:	ff ff 3f 
     fbc:	48 21 f7             	and    %rsi,%rdi
     fbf:	e8 00 00 00 00       	callq  fc4 <MPWorkspaceInit+0x24>
     fc4:	48 89 03             	mov    %rax,(%rbx)
     fc7:	5b                   	pop    %rbx
     fc8:	c3                   	retq   
     fc9:	bf 00 00 00 00       	mov    $0x0,%edi
     fce:	be 00 00 00 00       	mov    $0x0,%esi
     fd3:	ba 41 03 00 00       	mov    $0x341,%edx
     fd8:	b9 00 00 00 00       	mov    $0x0,%ecx
     fdd:	e8 00 00 00 00       	callq  fe2 <MPWorkspaceInit+0x42>
     fe2:	bf 00 00 00 00       	mov    $0x0,%edi
     fe7:	be 00 00 00 00       	mov    $0x0,%esi
     fec:	ba 42 03 00 00       	mov    $0x342,%edx
     ff1:	b9 00 00 00 00       	mov    $0x0,%ecx
     ff6:	e8 00 00 00 00       	callq  ffb <MPWorkspaceInit+0x5b>
     ffb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001000 <MPWorkspaceDestroy>:
    1000:	50                   	push   %rax
    1001:	48 85 ff             	test   %rdi,%rdi
    1004:	74 09                	je     100f <MPWorkspaceDestroy+0xf>
    1006:	48 8b 3f             	mov    (%rdi),%rdi
    1009:	58                   	pop    %rax
    100a:	e9 00 00 00 00       	jmpq   100f <MPWorkspaceDestroy+0xf>
    100f:	bf 00 00 00 00       	mov    $0x0,%edi
    1014:	be 00 00 00 00       	mov    $0x0,%esi
    1019:	ba 4b 03 00 00       	mov    $0x34b,%edx
    101e:	b9 00 00 00 00       	mov    $0x0,%ecx
    1023:	e8 00 00 00 00       	callq  1028 <MPWorkspaceDestroy+0x28>
