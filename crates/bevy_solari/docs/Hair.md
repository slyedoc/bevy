# Path Traced Hair & Skin in AAA games using Vulkan\
  
## Trace Primary Ray for Hair Only

- in a separate visibility buffer render pass
- ouptut depth, normal, primitiveIndex, and modelIndex
- when pixels are within hair aabb
- traceRay to hit the hair
    - trace with hair only mask (for perf)
    - compair hair depth with gbuffer depth to cull
      - (or adjust ray TMax to gbugger Depth)

  RTX Character Rendering SDK
  - Used BSDF (Bidirectional Scattering Distribution Function ) from Nvidia RTX Character
  Rendering SDK
  - Add multiplers to minimise the visual difference between raster and vs PT hair
  - Near-field Chiange BSDF vs Far-field BSDF (=used this for less noise)

  RayGen Shader

  - Max 6 bounces (+primary ray)
  - Sample new ray from Far-field BSDF: PrepareRay
   - New ray direction can be from R, TT, or TRT lob
  - Trace ray with new ray direction
    - CloestHitShader, returns the radiance
  - Shares the same PT loop as Non-hair
   - PrepareRay() implemenation is different

  ```
  bouseNum = hitDesc.isHair ? 6 : 2; // for hair, use max 6 extra bounces than the primary rays
  PrepareRay ( hitDesc.isHair, ...); // Sample new ray direction based on BSDF sampling, R/TT/TRT
  proability and roughnesses
  for ( uinit16_t bounceID = 0; bounceId, bounceNum; ++bounceId) { // iterate for multi bouces
    TraceRay(); // Trace rays to see if it hits something
    hitDesc = InitHitDesc( ... ); // decode hit material and shading result from the CHS
    pathDesc.radiance.xyz += half3( hitDesc.shading * float3( throughput ) ); // Accumate the
  shading
    boucneId = shouldTermitePath ? bounceNum : bounceId; / Ray miss (sky/emissive) no no thoughput
  then termitate
    ray = PrepareRay( hitDesc.isHair, ... ); // Decide which direction for the next journey
  }
  ```

  PrepareRay for Hairs

  - Unpack hair model ID and primitive ID for the payload
  - Decode hair matierla parametrs
  - Smaple BSDF -> new ray direction with PDF
  - if next ray goes though the hair
    - Offset ray origin

  ```
  PT_PREPARE_SHADING_SAMPLE(...) // Output hairSample
  RT_HAIR_SETUP_BSDF_MATERIAL( TBN, viewVectorLocal, lightVectorLocal, hairSample,
  hairMaterialInteraction, hairInteractionSuface );
  bool continueTrace = RTXCR_SampleFarFieldBcsdf( hairInteractionSurface,
  hairMaterialInteractioni, viewVectorLocal, h, lobeRandom, rand2, sampleDirection, bsdfSpecular,
  bsdfDiffuse, bsdfPdf );

  bsdfWeidht = bsdfDiffuse + bsdfSpecular;
  if (!contineTrace ) {
    throughput = _half3( 0 );
    return false;
  }

  bsdfWeight /= bsdfPdf;
  thoughput *= bsdfWeight;
  // new sample direction
  ray.direction.xyz = normalize(MatrixMult(TBN, sampleDirection ) );

  Offset Ray Origin

  - To avoid self-occlusion
  - Too big offset ( >  the strand width) can cause artifact
  - LSS hair is always back face culled in TraceRay ( = no inside-to-outside hit)

  // Refacction requires the ray offset to go in the opposite direction
  const bool transition =- dot (hitNormal, ray.direction.xyz ) <= 0.0f;
  const float3 offsetNormal = transition ? -hitNormal : hitNormal;
  // moving the ray origin slightly forward can avoid passing the head gemoetry when the hair //
  is attached to the head gemetry
  const float extraOffsetDistance = transition ? thichnes * 0.1f: 1e-5f;
  ray.origin = hitPos + offsetNormal * extraOffsetDistance;


  ## Offset Ray Origin Bug
  basiclly ray offst can test from instead the ead, so head gemoetry needs (back face culled)

  ## Ray Hit Normal Vector in Closest Hit Shader

  To Compute normal vector form hit position of hair:
   - Using the u(or x) parameter from 'hitAtttribeEXT vec3 baryCoord`
   - Compute Projected hit position along centerline ( = hisPosCenter)
   - Then, normal vector = hitPos - HitPosCenter

  ```
  hitAttributeEXT float uVal; // hit position along LSS midsection. [0 to 1]

  // returns the position of the two endpoints of LSS
  vec3 endCaps[2] = { g1_HitLSSPositionsNV[0], g1_hitLSSpositionNV[2] };

  // In object space
  const vec3 hitPosSurface = ( g1_objectRayDirectionEXT * g1_HitTEXT ) + g1_ObjectRayOriginEXT;

  // Interpolated position in midsection
  const vec3 hitPosCenter = lerp(endCaps[0], endCaps[1], uVal );
  vec3 rayHitNormalInObjectSpace = normlize(hitPosSurface - hitPosCenter  );

  // In world space
  float rayHitNormalInWorldSpace = mat3( g1_ObjectToWorldEXT) * rayHitNormalInObjectSpace;
  ```

  ## Smooth Tangent Normal

  - LSS Segments are not curved, in closer view, lighting looks flat for longer hairs
  - Interpolate tangent/normal from current segment to next segment
    - weight = u from barycetric
    - tangent = lerp( currentTanget, nextTangent, weight)

  ## Multi-Bounce Scattering

  - Intra-strand scattering is done by Far-field BSDF using R/TT/TRT
  - Intra-strand scattering is done by shooting more rays
  - Max 6 ( excluding primary ray) bounces for path-tracing loop

  1) Strand - Strand
  2) Strand -> Opaque geometry
  3) Opaque geometry -> Strand