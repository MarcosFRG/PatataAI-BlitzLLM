' Constantes del sistema optimizadas para 8GB RAM
Const MAX_TOKENS:Int = 10000 ' Original: 1M
Const EMBEDDING_SIZE:Int = 256 ' Original = 512 ... Mayor dimensi贸n para mejor calidad
Const MAX_CONTEXT:Int = 1024' Original = 2048 ... Contexto m谩s largo
Const LEARNING_RATE:Float = 0.001 ' M谩s estable para entrenamiento largo
Const MAX_RESPONSE_LENGTH:Int = 100 ' Respuestas m谩s largas
Const IS_LEARNING:Int = 1
Const TEMPERATURE:Float = 0.2
Const MEMORY_DECAY:Float = 0.97 ' Olvido m谩s lento
Const BATCH_SIZE:Int = 4 ' Recomendado: 4 o 8
Const CHECKPOINT_EVERY:Int = 1000 ' Autoguardado cada 1000 interacciones
Const FFN_HIDDEN_SIZE:Int = 2048 ' Tama帽o de la capa oculta FFN
Const FFN_DROPOUT:Float = 0.1 ' Tasa de dropout para FFN
Const FFN_ACTIVATION_SCALE:Float = 0.1 ' Escala para activaci贸n GELU

' Tipos de tokens especiales
Const TK_UNKNOWN:Int = 0
Const TK_START:Int = 1
Const TK_END:Int = 2
Const TK_RES:Int = 3
Const TK_PROMPT:Int = 4
Const TK_NEWLINE:Int = 5

' Estructura para pares BPE
Type TBpePair
    Field left:Int
    Field right:Int
    Field count:Int
End Type

Type TSparseMatrix
    Field rows:Int
    Field cols:Int
    Field data:Float[,] ' [index, 0=row, 1=col, 2=value]
    
    Method New(rows:Int, cols:Int)
        Self.rows = rows
        Self.cols = cols
        data = New Float[rows*cols/10, 3] ' Asume 10% de densidad
    End Method
    
    Method SetValue(row:Int, col:Int, value:Float)
        For idx_data:Int = 0 Until data.Length
            If data[idx_data, 0] = 0 And data[idx_data, 1] = 0
                data[idx_data, 0] = row
                data[idx_data, 1] = col
                data[idx_data, 2] = value
                Return
            EndIf
        Next
        ' Si llegamos aqu铆, necesitamos redimensionar
        Local newData:Float[,] = New Float[data.Length*2, 3]
        For igg:Int = 0 Until data.Length
            For jaa:Int = 0 Until 3
                newData[igg,jaa] = data[igg,jaa]
            Next
        Next
        data = newData
        SetValue(row, col, value)
    End Method
    
    Method GetValue:Float(row:Int, col:Int)
        For idx_search:Int = 0 Until data.Length
            If data[idx_search, 0] = row And data[idx_search, 1] = col
                Return data[idx_search, 2]
            EndIf
        Next
        Return 0.0
    End Method
End Type

Type THierarchicalMemory
    Field shortTerm:Float[]
    Field mediumTerm:Float[]
    Field longTerm:Float[]
    Field updateCounter:Int
    
    Method New()
        shortTerm = New Float[EMBEDDING_SIZE]
        mediumTerm = New Float[EMBEDDING_SIZE]
        longTerm = New Float[EMBEDDING_SIZE]
    End Method
    
    Method UpdateMemory(context:Float[])
        ' Actualizaci贸n en capas
        For emb_mem:Int = 0 Until EMBEDDING_SIZE
            shortTerm[emb_mem] = 0.9 * shortTerm[emb_mem] + 0.1 * context[emb_mem]
            
            If updateCounter Mod 10 = 0
                mediumTerm[emb_mem] = 0.95 * mediumTerm[emb_mem] + 0.05 * shortTerm[emb_mem]
            EndIf
            
            If updateCounter Mod 100 = 0
                longTerm[emb_mem] = 0.99 * longTerm[emb_mem] + 0.01 * mediumTerm[emb_mem]
            EndIf
        Next
        
        updateCounter :+ 1
    End Method
    
    Method GetContext:Float[]()
        Local result_mem:Float[] = New Float[EMBEDDING_SIZE]
        For emb_result:Int = 0 Until EMBEDDING_SIZE
            result_mem[emb_result] = 0.6*shortTerm[emb_result] + 0.3*mediumTerm[emb_result] + 0.1*longTerm[emb_result]
        Next
        Return result_mem
    End Method
    
    Method Clear()
        For emb_clear:Int = 0 Until EMBEDDING_SIZE
            shortTerm[emb_clear] = 0
            mediumTerm[emb_clear] = 0
            longTerm[emb_clear] = 0
        Next
        updateCounter = 0
    End Method
End Type

Type TFeedForwardNetwork
    Field weights1:Float[,] ' EMBEDDING_SIZE x FFN_HIDDEN_SIZE
    Field weights2:Float[,] ' FFN_HIDDEN_SIZE x EMBEDDING_SIZE
    Field bias1:Float[]     ' FFN_HIDDEN_SIZE
    Field bias2:Float[]     ' EMBEDDING_SIZE
    
    Method New()
        weights1 = New Float[EMBEDDING_SIZE, FFN_HIDDEN_SIZE]
        weights2 = New Float[FFN_HIDDEN_SIZE, EMBEDDING_SIZE]
        bias1 = New Float[FFN_HIDDEN_SIZE]
        bias2 = New Float[EMBEDDING_SIZE]
        
        InitializeWeights()
    End Method
    
    Method InitializeWeights()
        ' Inicializaci贸n Xavier/Glorot para la primera capa
        Local scale1:Float = Sqr(2.0 / (EMBEDDING_SIZE + FFN_HIDDEN_SIZE))
        For i_ffn1:Int = 0 Until EMBEDDING_SIZE
            For j_ffn1:Int = 0 Until FFN_HIDDEN_SIZE
                weights1[i_ffn1, j_ffn1] = Rnd(-scale1, scale1)
            Next
        Next
        
        ' Inicializaci贸n Xavier/Glorot para la segunda capa
        Local scale2:Float = Sqr(2.0 / (FFN_HIDDEN_SIZE + EMBEDDING_SIZE))
        For i_ffn2:Int = 0 Until FFN_HIDDEN_SIZE
            For j_ffn2:Int = 0 Until EMBEDDING_SIZE
                weights2[i_ffn2, j_ffn2] = Rnd(-scale2, scale2)
            Next
        Next
        
        ' Inicializaci贸n de biases
        For b1_idx:Int = 0 Until FFN_HIDDEN_SIZE
            bias1[b1_idx] = Rnd(-0.01, 0.01)
        Next
        
        For b2_idx:Int = 0 Until EMBEDDING_SIZE
            bias2[b2_idx] = Rnd(-0.01, 0.01)
        Next
    End Method
    
    Method Process:Float[](input:Float[])
        ' Capa oculta
        Local hidden:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For h_idx:Int = 0 Until FFN_HIDDEN_SIZE
            hidden[h_idx] = bias1[h_idx]
            
            For in_idx:Int = 0 Until EMBEDDING_SIZE
                hidden[h_idx] :+ input[in_idx] * weights1[in_idx, h_idx]
            Next
            
            ' Activaci贸n GELU aproximada
            hidden[h_idx] = GELU(hidden[h_idx])
            
            ' Aplicar dropout durante el entrenamiento
            If IS_LEARNING And Rnd() < FFN_DROPOUT
                hidden[h_idx] = 0
            EndIf
        Next
        
        ' Capa de salida
        Local output:Float[] = New Float[EMBEDDING_SIZE]
        
        For out_idx:Int = 0 Until EMBEDDING_SIZE
            output[out_idx] = bias2[out_idx]
            
            For h_out_idx:Int = 0 Until FFN_HIDDEN_SIZE
                output[out_idx] :+ hidden[h_out_idx] * weights2[h_out_idx, out_idx]
            Next
        Next
        
        Return output
    End Method
    
    Method GELU:Float(x:Float)
        ' Aproximaci贸n de la funci贸n GELU
        Return 0.5 * x * (1 + Tanh(FFN_ACTIVATION_SCALE * (x + 0.044715 * x * x * x)))
    End Method
    
    Method Backpropagate(input:Float[], gradOutput:Float[], gradInput:Float[] Var, gradWeights1:Float[,] Var, gradWeights2:Float[,] Var, gradBias1:Float[] Var, gradBias2:Float[] Var)
        ' Forward pass para obtener activaciones
        Local hidden:Float[] = New Float[FFN_HIDDEN_SIZE]
        Local hiddenPreActivation:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For h_bp_idx:Int = 0 Until FFN_HIDDEN_SIZE
            hiddenPreActivation[h_bp_idx] = bias1[h_bp_idx]
            
            For in_bp_idx:Int = 0 Until EMBEDDING_SIZE
                hiddenPreActivation[h_bp_idx] :+ input[in_bp_idx] * weights1[in_bp_idx, h_bp_idx]
            Next
            
            hidden[h_bp_idx] = GELU(hiddenPreActivation[h_bp_idx])
        Next
        
        ' Gradientes de la segunda capa
        For out_bp_idx:Int = 0 Until EMBEDDING_SIZE
            gradBias2[out_bp_idx] :+ gradOutput[out_bp_idx] * LEARNING_RATE
            
            For h_bp2_idx:Int = 0 Until FFN_HIDDEN_SIZE
                gradWeights2[h_bp2_idx, out_bp_idx] :+ hidden[h_bp2_idx] * gradOutput[out_bp_idx] * LEARNING_RATE
            Next
        Next
        
        ' Gradientes de la capa oculta
        Local gradHidden:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For h_grad_idx:Int = 0 Until FFN_HIDDEN_SIZE
            Local sum:Float = 0
            For out_grad_idx:Int = 0 Until EMBEDDING_SIZE
                sum :+ gradOutput[out_grad_idx] * weights2[h_grad_idx, out_grad_idx]
            Next
            
            ' Derivada de GELU
            Local x:Float = hiddenPreActivation[h_grad_idx]
            Local geluDeriv:Float = 0.5 * Tanh(FFN_ACTIVATION_SCALE * (x + 0.044715 * x * x * x)) + ..
                                  (0.5 * x * FFN_ACTIVATION_SCALE * (1 + 3 * 0.044715 * x * x)) / ..
                                  (Cosh(FFN_ACTIVATION_SCALE * (x + 0.044715 * x * x * x)) ^ 2)
            
            gradHidden[h_grad_idx] = sum * geluDeriv
        Next
        
        ' Gradientes de la primera capa
        For h_grad1_idx:Int = 0 Until FFN_HIDDEN_SIZE
            gradBias1[h_grad1_idx] :+ gradHidden[h_grad1_idx] * LEARNING_RATE
            
            For in_grad_idx:Int = 0 Until EMBEDDING_SIZE
                gradWeights1[in_grad_idx, h_grad1_idx] :+ input[in_grad_idx] * gradHidden[h_grad1_idx] * LEARNING_RATE
            Next
        Next
        
        ' Gradiente de entrada (si es necesario)
        For in_gradin_idx:Int = 0 Until EMBEDDING_SIZE
            gradInput[in_gradin_idx] = 0
            
            For h_gradin_idx:Int = 0 Until FFN_HIDDEN_SIZE
                gradInput[in_gradin_idx] :+ gradHidden[h_gradin_idx] * weights1[in_gradin_idx, h_gradin_idx]
            Next
        Next
    End Method
End Type

Type TAdvancedLLM
    Field tokenDB:String[]
    Field tokenCounts:Int[]
    Field embeddingWeights:Float[,]
    Field attentionWeights:Float[,,]
    Field memory:THierarchicalMemory
    Field totalInteractions:Long
    Field learningEnabled:Int = IS_LEARNING
    Field enginePrompt:String
    Field conversationHistory:String[]
    Field currentContext:Float[]
    Field temperature:Float = TEMPERATURE
    Field bpePairs:TBpePair[]
    Field batchBuffer:TList
    Field lastCheckpoint:Int
    Field ffn:TFeedForwardNetwork
    
    Method New()
        tokenDB = New String[MAX_TOKENS]
        tokenCounts = New Int[MAX_TOKENS]
        embeddingWeights = New Float[MAX_TOKENS, EMBEDDING_SIZE]
        attentionWeights = New Float[8, EMBEDDING_SIZE, EMBEDDING_SIZE]
        conversationHistory = New String[0]
        memory = New THierarchicalMemory
        currentContext = New Float[EMBEDDING_SIZE]
        bpePairs = New TBpePair[0]
        batchBuffer = CreateList()
        ffn = New TFeedForwardNetwork
        
        InitializeSpecialTokens()
        InitializeWeights()
        ClearContext()
    End Method
    
    Method InitializeSpecialTokens()
        tokenDB[TK_UNKNOWN] = "[UNK]"
        tokenDB[TK_START] = "[START]"
        tokenDB[TK_END] = "[END]"
        tokenDB[TK_RES] = "[RES]"
        tokenDB[TK_PROMPT] = "[PROMPT]"
        tokenDB[TK_NEWLINE] = "~n"
        
        For token_init:Int = 0 To 5
            tokenCounts[token_init] = 1
        Next
    End Method
    
    Method InitializeWeights()
        ' Inicializaci贸n Xavier/Glorot para embeddings
        Local scale_emb:Float = Sqr(6.0 / (MAX_TOKENS + EMBEDDING_SIZE))
        For token_emb:Int = 0 To MAX_TOKENS-1
            For emb_emb:Int = 0 To EMBEDDING_SIZE-1
                embeddingWeights[token_emb, emb_emb] = Rnd(-scale_emb, scale_emb)
            Next
        Next
        
        ' Inicializaci贸n de atenci贸n con transformaciones ortogonales
        For head_att:Int = 0 To 7
            For row_att:Int = 0 To EMBEDDING_SIZE-1
                For col_att:Int = 0 To EMBEDDING_SIZE-1
                    If row_att = col_att
                        attentionWeights[head_att, row_att, col_att] = 0.1 * Rnd(0.9, 1.1)
                    Else
                        attentionWeights[head_att, row_att, col_att] = 0.01 * Rnd(-1, 1)
                    EndIf
                Next
            Next
        Next
    End Method
    
    Method SetTemperature(temp:Float)
        temperature = Max(0.1, Min(2.0, temp))
    End Method
    
    Method SetEnginePrompt(prompt:String)
        enginePrompt = prompt
        ' Solo a帽adir al historial si el prompt no est谩 vac铆o
        If prompt <> ""
            If conversationHistory.Length = 0
                conversationHistory :+ ["System: " + prompt]
            Else
                conversationHistory[0] = "System: " + prompt
            EndIf
        Else
            ' Si el prompt est谩 vac铆o, limpiar el historial si el primer elemento es un prompt del sistema
            If conversationHistory.Length > 0 And conversationHistory[0].StartsWith("System:")
                conversationHistory = conversationHistory[1..]
            EndIf
        EndIf
    End Method
    
    Method AddToHistory(message:String, isUser:Int)
        If isUser
            conversationHistory :+ ["User: " + message]
        Else
            conversationHistory :+ ["AI: " + message]
        EndIf
        
        ' Limpieza adaptativa del historial
        If conversationHistory.Length > 24
            ' Mantener siempre el prompt del sistema si existe
            Local newHistory_hist:String[]
            If enginePrompt <> "" And conversationHistory.Length > 0 And conversationHistory[0].StartsWith("System:")
                newHistory_hist :+ [conversationHistory[0]]
            End If
            
            Local importanceScores_hist:Float[] = New Float[conversationHistory.Length]
            Local totalScore_hist:Float = 0
            
            ' Calcular puntuaciones de importancia
            For hist_calc:Int = 0 Until conversationHistory.Length
                ' Saltar el prompt del sistema si ya lo hemos a帽adido
                If enginePrompt <> "" And hist_calc = 0 And conversationHistory[0].StartsWith("System:") Then Continue
                
                importanceScores_hist[hist_calc] = CalculateImportance(conversationHistory[hist_calc])
                totalScore_hist :+ importanceScores_hist[hist_calc]
            Next
            
            ' Seleccionar los m谩s importantes
            For hist_sel:Int = 0 Until conversationHistory.Length
                ' Saltar el prompt del sistema si ya lo hemos a帽adido
                If enginePrompt <> "" And hist_sel = 0 And conversationHistory[0].StartsWith("System:") Then Continue
                
                If importanceScores_hist[hist_sel] > totalScore_hist/conversationHistory.Length Or ..
                   hist_sel >= conversationHistory.Length-4 ' Mantener 煤ltimos 4 mensajes
                    newHistory_hist :+ [conversationHistory[hist_sel]]
                EndIf
            Next
            
            conversationHistory = newHistory_hist
        EndIf
    End Method
    
    Method CalculateImportance:Float(message:String)
        Local score_imp:Float = 0.2 ' Base
        score_imp :+ 0.0005 * message.Length ' Mensajes largos m谩s importantes
        
        ' Ponderar por contenido
        If message.Find("?") >= 0 Then score_imp :+ 0.3 ' Preguntas
        If message.Find("!") >= 0 Then score_imp :+ 0.2 ' 脙鈥皀fasis
        If message.Find("http://") >=0 Or message.Find("https://") >=0 Then score_imp :+ 0.5 ' URLs
        
        ' Mensajes del sistema son m谩s importantes
        If message.StartsWith("System:") Then score_imp :+ 1.0
        
        Return Min(2.0, score_imp)
    End Method
    
    Method ClearContext()
        memory.Clear()
        ' Solo mantener el prompt si no est谩 vac铆o
        If enginePrompt <> ""
            conversationHistory = ["System: " + enginePrompt]
        Else
            conversationHistory = New String[0]
        EndIf
    End Method
    
    Method EnableLearning(enable:Int)
        learningEnabled = enable
    End Method
    
    ' Tokenizaci贸n mejorada con BPE b谩sico
Method Tokenizar:String[](texto:String)
    texto = texto.ToLower()
    
    ' Filtro seguro para caracteres:
    Local tokens:String[]
    Local i:Int = 0
    
    While i < texto.Length
        Local charCode:Int = texto[i]
        Local c:String = Chr(charCode)
        
        ' Identificar letras (usando c贸digos ASCII)
        If (charCode >= Asc("a") And charCode <= Asc("z")) Or charCode = Asc("帽")
            Local palabra:String
            While i < texto.Length And ( (texto[i] >= Asc("a") And texto[i] <= Asc("z")) Or texto[i] = Asc("'") Or texto[i] = Asc("帽") )
                palabra :+ Chr(texto[i])
                i :+ 1
            Wend
            tokens :+ [palabra]
        Else
            ' Manejar espacios y puntuaci贸n
            Select charCode
                Case Asc(" "), Asc("~t"), Asc("~n"), Asc("."), Asc(","), Asc("!"), Asc("?"), Asc("驴"), Asc("隆")
                    tokens :+ [c]
            End Select
            i :+ 1
        EndIf
    Wend
    
    Return tokens
End Method
    
    Method ObtenerOCrearID:Int(token:String)
        ' Primero revisar tokens especiales
        Select token
            Case "[UNK]" Return TK_UNKNOWN
            Case "[START]" Return TK_START
            Case "[END]" Return TK_END
            Case "[RES]" Return TK_RES
            Case "[PROMPT]" Return TK_PROMPT
            Case "~n" Return TK_NEWLINE
        End Select
        
        ' B煤squeda optimizada con cach脙漏
        Local cache_tok:TMap = CreateMap()
        Local cachedID_tok:Object = cache_tok.ValueForKey(token)
        If cachedID_tok <> Null Then Return Int(String(cachedID_tok))
        
        ' B煤squeda en zona no especial
        For tokenID_tok:Int = 6 To MAX_TOKENS-1
            If tokenDB[tokenID_tok] = token
                cache_tok.Insert(token, String(tokenID_tok))
                Return tokenID_tok
            EndIf
        Next
        
        ' Creaci贸n de nuevo token
        For newID_tok:Int = 6 To MAX_TOKENS-1
            If tokenDB[newID_tok] = Null Or tokenDB[newID_tok] = ""
                tokenDB[newID_tok] = token
                tokenCounts[newID_tok] = 1
                
                ' Inicializaci贸n adaptativa basada en tokens similares
                Local similarFound_tok:Int = False
                For emb_tok:Int = 0 To EMBEDDING_SIZE-1
                    Local sum_tok:Float = 0
                    Local count_tok:Int = 0
                    
                    ' Promedio de embeddings de sub-tokens (para palabras compuestas)
                    For subPos_tok:Int = 0 Until token.Length-1
                        Local subToken_tok:String = token[subPos_tok..subPos_tok+1]
                        Local subID_tok:Int = ObtenerOCrearID(subToken_tok)
                        If subID_tok <> TK_UNKNOWN
                            sum_tok :+ embeddingWeights[subID_tok, emb_tok]
                            count_tok :+ 1
                        EndIf
                    Next
                    
                    If count_tok > 0
                        embeddingWeights[newID_tok, emb_tok] = sum_tok / count_tok
                        similarFound_tok = True
                    Else
                        Local scale_tok:Float = Sqr(2.0 / (newID_tok+1 + EMBEDDING_SIZE))
                        embeddingWeights[newID_tok, emb_tok] = Rnd(-scale_tok, scale_tok)
                    EndIf
                Next
                
                cache_tok.Insert(token, String(newID_tok))
                Return newID_tok
            EndIf
        Next
        
        Return TK_UNKNOWN
    End Method
    
    Method MultiHeadAttention:Float[](input:Float[], headID:Int, useMemory:Int = True)
    ' Validar par谩metros de entrada
    If headID < 0 Or headID > 7 Then Return New Float[EMBEDDING_SIZE] ' 8 cabezas (0-7)
    If input.Length <> EMBEDDING_SIZE Then Return New Float[EMBEDDING_SIZE]
    
    Local output_att:Float[] = New Float[EMBEDDING_SIZE]
    Local key_att:Float[] = New Float[EMBEDDING_SIZE]
    Local value_att:Float[] = New Float[EMBEDDING_SIZE]
    
    ' Proyecciones Q, K, V con verificaci贸n de l铆mites
    For Local embFrom_att:Int = 0 Until EMBEDDING_SIZE
        For Local embTo_att:Int = 0 Until EMBEDDING_SIZE
            ' Calcular offsets seguros para los pesos de atenci贸n
            Local weightOffset_att:Int = headID * 3
            If weightOffset_att + 2 >= 8 Then Continue ' Solo tenemos 8 cabezas (0-7)
            
            ' Proyecci贸n Key (K)
            If weightOffset_att < 8 And embFrom_att < EMBEDDING_SIZE And embTo_att < EMBEDDING_SIZE
                key_att[embTo_att] :+ input[embFrom_att] * attentionWeights[weightOffset_att, embFrom_att, embTo_att]
            EndIf
            
            ' Proyecci贸n Value (V)
            If weightOffset_att + 1 < 8 And embFrom_att < EMBEDDING_SIZE And embTo_att < EMBEDDING_SIZE
                value_att[embTo_att] :+ input[embFrom_att] * attentionWeights[weightOffset_att+1, embFrom_att, embTo_att]
            EndIf
            
            ' Proyecci贸n Output (Q)
            If weightOffset_att + 2 < 8 And embFrom_att < EMBEDDING_SIZE And embTo_att < EMBEDDING_SIZE
                output_att[embTo_att] :+ input[embFrom_att] * attentionWeights[weightOffset_att+2, embFrom_att, embTo_att]
            EndIf
        Next
    Next
    
    ' Aplicar memoria jer谩rquica si est谩 activado
    If useMemory
        Local memContext_att:Float[] = memory.GetContext()
        For Local emb_mematt:Int = 0 Until EMBEDDING_SIZE
            output_att[emb_mematt] :+ 0.4 * memContext_att[emb_mematt] * output_att[emb_mematt]
        Next
    EndIf
    
    ' Softmax con temperatura y estabilidad num脙漏rica
    Local maxVal_att:Float = output_att[0]
    For Local emb_max:Int = 1 Until EMBEDDING_SIZE
        If output_att[emb_max] > maxVal_att Then maxVal_att = output_att[emb_max]
    Next
    
    Local sumExp_att:Float = 0
    For Local emb_sm:Int = 0 Until EMBEDDING_SIZE
        output_att[emb_sm] = Exp((output_att[emb_sm] - maxVal_att)/temperature)
        sumExp_att :+ output_att[emb_sm]
    Next
    
    If sumExp_att > 0
        For Local emb_norm:Int = 0 Until EMBEDDING_SIZE
            output_att[emb_norm] :/ sumExp_att
            output_att[emb_norm] :* value_att[emb_norm] ' Multiplicar por valores
        Next
    EndIf
    
    ' Aplicar FFN
    output_att = ffn.Process(output_att)
    
    Return output_att
End Method
    
    Method ProcesarEntrada:Float[](tokens:Int[])
        Local context_proc:Float[] = New Float[EMBEDDING_SIZE]
        Local newMemory_proc:Float[] = New Float[EMBEDDING_SIZE]
        
        ' Procesar en lotes para mejor eficiencia
        For batchStart_proc:Int = 0 Until tokens.Length Step BATCH_SIZE
            Local batchEnd_proc:Int = Min(batchStart_proc + BATCH_SIZE, tokens.Length)
            
            ' Procesar cada token en el lote actual
            For tokenPos_proc:Int = batchStart_proc Until batchEnd_proc
                Local tokenID_proc:Int = tokens[tokenPos_proc]
                Local tokenEmbedding_proc:Float[] = New Float[EMBEDDING_SIZE]
                
                For emb_embed:Int = 0 To EMBEDDING_SIZE-1
                    tokenEmbedding_proc[emb_embed] = embeddingWeights[tokenID_proc, emb_embed]
                Next
                
                ' Atenci贸n en dos fases con diferentes cabezas
                For phase_proc:Int = 0 To 1
                    Local headStart_proc:Int = phase_proc * 4 ' Usamos 4 cabezas por fase
                    
                    For headOffset_proc:Int = 0 To 3
                        Local currentHead_proc:Int = headStart_proc + headOffset_proc
                        Local attended_proc:Float[] = MultiHeadAttention(tokenEmbedding_proc, currentHead_proc, phase_proc=1)
                        
                        ' Actualizar contexto y memoria
                        For emb_update:Int = 0 To EMBEDDING_SIZE-1
                            If phase_proc = 0
                                context_proc[emb_update] :+ attended_proc[emb_update] * 0.5
                                newMemory_proc[emb_update] :+ attended_proc[emb_update] * 0.05
                            Else
                                context_proc[emb_update] :+ attended_proc[emb_update] * 0.5
                            EndIf
                        Next
                    Next
                Next
                
                ' Contabilizar el token
                tokenCounts[tokenID_proc] :+ 1
            Next
            
            ' Normalizaci贸n intermedia del contexto
            Local norm_proc:Float = 0
            For emb_normproc:Int = 0 To EMBEDDING_SIZE-1
                norm_proc :+ context_proc[emb_normproc] * context_proc[emb_normproc]
            Next
            norm_proc = Sqr(norm_proc)
            
            If norm_proc > 0
                For emb_normproc2:Int = 0 To EMBEDDING_SIZE-1
                    context_proc[emb_normproc2] :/ norm_proc
                Next
            EndIf
        Next
        
        ' Actualizar memoria jer谩rquica
        memory.UpdateMemory(newMemory_proc)
        
        ' Normalizaci贸n final
        Local norm_final:Float = 0
        For emb_final:Int = 0 To EMBEDDING_SIZE-1
            norm_final :+ context_proc[emb_final] * context_proc[emb_final]
        Next
        norm_final = Sqr(norm_final)
        
        If norm_final > 0
            For emb_final2:Int = 0 To EMBEDDING_SIZE-1
                context_proc[emb_final2] :/ norm_final
                currentContext[emb_final2] = 0.8 * currentContext[emb_final2] + 0.2 * context_proc[emb_final2]
            Next
        EndIf
        
        Return currentContext
    End Method
    
    Method Backpropagate(inputIDs:Int[], outputIDs:Int[], context:Float[])
        ' Gradientes para embeddings
        Local gradEmbed_back:Float[,] = New Float[MAX_TOKENS, EMBEDDING_SIZE]
        
        ' Calcular gradientes de salida
        For outPos_back:Int = 0 Until outputIDs.Length
            Local tokenID_back:Int = outputIDs[outPos_back]
            For emb_back:Int = 0 To EMBEDDING_SIZE-1
                Local error_back:Float = context[emb_back] - embeddingWeights[tokenID_back, emb_back]
                gradEmbed_back[tokenID_back, emb_back] :+ LEARNING_RATE * error_back / outputIDs.Length
            Next
        Next
        
        ' Retropropagaci贸n a trav脙漏s de atenci贸n y FFN
        Local gradAttention_back:Float[,,] = New Float[8, EMBEDDING_SIZE, EMBEDDING_SIZE]
        Local gradFFNInput_back:Float[] = New Float[EMBEDDING_SIZE]
        
        For inPos_back:Int = 0 Until inputIDs.Length
            Local tokenID_back2:Int = inputIDs[inPos_back]
            Local tokenEmbed_back:Float[] = New Float[EMBEDDING_SIZE]
            
            For emb_embedback:Int = 0 To EMBEDDING_SIZE-1
                tokenEmbed_back[emb_embedback] = embeddingWeights[tokenID_back2, emb_embedback]
            Next
            
            For headID_back:Int = 0 To 5 Step 2
                ' Calcular gradientes para esta cabeza
                For embFrom_back:Int = 0 To EMBEDDING_SIZE-1
                    For embTo_back:Int = 0 To EMBEDDING_SIZE-1
                        Local grad_back:Float = tokenEmbed_back[embFrom_back] * context[embTo_back] * LEARNING_RATE * 0.001
                        gradAttention_back[headID_back, embFrom_back, embTo_back] :+ grad_back
                        gradAttention_back[headID_back+1, embFrom_back, embTo_back] :+ grad_back * 0.5
                    Next
                Next
            Next
            
            ' Retropropagar a trav脙漏s del FFN
            Local gradFFNWeights1_back:Float[,] = New Float[EMBEDDING_SIZE, FFN_HIDDEN_SIZE]
            Local gradFFNWeights2_back:Float[,] = New Float[FFN_HIDDEN_SIZE, EMBEDDING_SIZE]
            Local gradFFNBias1_back:Float[] = New Float[FFN_HIDDEN_SIZE]
            Local gradFFNBias2_back:Float[] = New Float[EMBEDDING_SIZE]
            
            ffn.Backpropagate(tokenEmbed_back, context, gradFFNInput_back, gradFFNWeights1_back, gradFFNWeights2_back, gradFFNBias1_back, gradFFNBias2_back)
            
            ' Aplicar gradientes del FFN
            For i_ffnback:Int = 0 Until EMBEDDING_SIZE
                For j_ffnback:Int = 0 Until FFN_HIDDEN_SIZE
                    ffn.weights1[i_ffnback, j_ffnback] :+ gradFFNWeights1_back[i_ffnback, j_ffnback]
                Next
            Next
            
            For i_ffnback2:Int = 0 Until FFN_HIDDEN_SIZE
                For j_ffnback2:Int = 0 Until EMBEDDING_SIZE
                    ffn.weights2[i_ffnback2, j_ffnback2] :+ gradFFNWeights2_back[i_ffnback2, j_ffnback2]
                Next
            Next
            
            For b_ffnback:Int = 0 Until FFN_HIDDEN_SIZE
                ffn.bias1[b_ffnback] :+ gradFFNBias1_back[b_ffnback]
            Next
            
            For b_ffnback2:Int = 0 Until EMBEDDING_SIZE
                ffn.bias2[b_ffnback2] :+ gradFFNBias2_back[b_ffnback2]
            Next
        Next
        
        ' Aplicar actualizaciones con decaimiento
        For token_upd:Int = 0 To MAX_TOKENS-1
            For emb_upd:Int = 0 To EMBEDDING_SIZE-1
                embeddingWeights[token_upd, emb_upd] :+ gradEmbed_back[token_upd, emb_upd]
                embeddingWeights[token_upd, emb_upd] :* 0.9999 ' Decaimiento de pesos
            Next
        Next
        
        For head_upd:Int = 0 To 7
            For embFrom_upd:Int = 0 To EMBEDDING_SIZE-1
                For embTo_upd:Int = 0 To EMBEDDING_SIZE-1
                    attentionWeights[head_upd, embFrom_upd, embTo_upd] :+ gradAttention_back[head_upd, embFrom_upd, embTo_upd]
                    attentionWeights[head_upd, embFrom_upd, embTo_upd] :* 0.999 ' Decaimiento de pesos
                Next
            Next
        Next
    End Method
    
    Method EntrenarLLM:Int(pregunta:String, respuesta:String, verbose:Int = False)
        Try
            ' Validaci贸n de entrada
            If pregunta.Trim().Length = 0 Or respuesta.Trim().Length = 0
                If verbose Then Print "Error: Pregunta o respuesta vac铆a"
                Return False
            EndIf
            
            ' Tokenizaci贸n mejorada
            Local inputTokens_train:String[] = Self.Tokenizar(pregunta)
            Local outputTokens_train:String[] = Self.Tokenizar(respuesta)
            
            If inputTokens_train.Length = 0 Or outputTokens_train.Length = 0
                If verbose Then Print "Error: No se generaron tokens v谩lidos"
                Return False
            EndIf
            
            ' Modo verbose
            If verbose
                Print "Entrenando con:"
                Print "  Pregunta ("+inputTokens_train.Length+" tokens): " + pregunta
                Print "  Respuesta ("+outputTokens_train.Length+" tokens): " + respuesta
                Print "  Tokens 煤nicos en DB: " + Self.ContarTokensActivos()
            EndIf
            
            ' Convertir a IDs
            Local inputIDs_train:Int[] = New Int[inputTokens_train.Length]
            Local outputIDs_train:Int[] = New Int[outputTokens_train.Length]
            
            For tokenPos_train:Int = 0 To inputTokens_train.Length-1
                inputIDs_train[tokenPos_train] = ObtenerOCrearID(inputTokens_train[tokenPos_train])
            Next
            
            For tokenPos_train2:Int = 0 To outputTokens_train.Length-1
                outputIDs_train[tokenPos_train2] = ObtenerOCrearID(outputTokens_train[tokenPos_train2])
            Next
            
            ' Procesamiento y aprendizaje
            Local context_train:Float[] = ProcesarEntrada(inputIDs_train)
            If learningEnabled
                Backpropagate(inputIDs_train, outputIDs_train, context_train)
                totalInteractions :+ 1
                
                ' Auto-guardado peri贸dico
                If totalInteractions - lastCheckpoint >= CHECKPOINT_EVERY
                    Local nombreArchivo_train:String = "checkpoint_"+totalInteractions+".llm"
                    GuardarModelo(nombreArchivo_train, False)
                    lastCheckpoint = totalInteractions
                    If verbose Then Print "Checkpoint guardado: " + nombreArchivo_train
                EndIf
            EndIf
            
            Return True
            
        Catch err_train:Object
            Print "Error cr韙ico en EntrenarLLM: " + err_train.ToString()
            Return False
        End Try
    End Method
    
Method GenerarRespuesta:String(input:String)
    AddToHistory(input, True)
    
    ' 1. Limitar contexto
    Local histCount:Int = Min(conversationHistory.Length, 6)
    Local recentContext:String[histCount]
    For Local histIdx:Int = 0 Until histCount
        recentContext[histIdx] = conversationHistory[conversationHistory.Length - histCount + histIdx]
    Next
    
    ' 2. Tokenizaci髇 segura
    Local processedTokens:String[] = Tokenizar(" ".Join(recentContext))
    Local tokenIDs:Int[processedTokens.Length]
    
    For Local tokenPos:Int = 0 Until processedTokens.Length
        tokenIDs[tokenPos] = ObtenerOCrearID(processedTokens[tokenPos])
    Next
    
    ' 3. Procesamiento con memoria
    Local contextVector:Float[EMBEDDING_SIZE]
    contextVector = ProcesarEntrada(tokenIDs)
    
    ' 4. Generaci髇 de respuesta
    Local generatedText:String
    Local currentToken:Int = TK_RES
    Local generatedCount:Int = 0
    Local recentTokens:Int[] = [0, 0, 0] ' Buffer de 鷏timos 3 tokens
    
    While generatedCount < MAX_RESPONSE_LENGTH
        ' 4.1 Calcular scores
        Local tokenScores:Float[MAX_TOKENS]
        Local totalScore:Float = 0
        Local bestToken:Int = -1
        Local highestScore:Float = -1
        
        For Local tkID:Int = 0 Until MAX_TOKENS
            If tokenDB[tkID] And tkID <> currentToken
                ' Penalizaci髇 por repetici髇
                Local penalty:Float = 1.0
                For Local bufIdx:Int = 0 Until recentTokens.Length
                    If recentTokens[bufIdx] = tkID Then penalty = 0.3
                Next
                
                ' C醠culo de score
                tokenScores[tkID] = 0
                For Local embIdx:Int = 0 Until EMBEDDING_SIZE
                    tokenScores[tkID] :+ embeddingWeights[currentToken, embIdx] * embeddingWeights[tkID, embIdx]
                Next
                
                tokenScores[tkID] :* (1.0 + 0.1 * Log(tokenCounts[tkID]+10)) * penalty
                
                ' Ajuste por contexto
                For Local ctxIdx:Int = 0 Until EMBEDDING_SIZE
                    tokenScores[tkID] :+ 0.5 * contextVector[ctxIdx] * embeddingWeights[tkID, ctxIdx]
                Next
                
                ' Aplicar temperatura
                tokenScores[tkID] = Exp(tokenScores[tkID]/temperature)
                totalScore :+ tokenScores[tkID]
                
                If tokenScores[tkID] > highestScore
                    highestScore = tokenScores[tkID]
                    bestToken = tkID
                EndIf
            EndIf
        Next
        
        ' 4.2 Selecci髇 de token
        If bestToken = -1 Then Exit
        
        Local selectedToken:Int = bestToken
        If temperature > 0.1 And totalScore > 0
            Local randomVal:Float = Rnd() * totalScore
            Local scoreSum:Float = 0
            For Local tk:Int = 0 Until MAX_TOKENS
                If tokenScores[tk] > 0
                    scoreSum :+ tokenScores[tk]
                    If scoreSum >= randomVal
                        selectedToken = tk
                        Exit
                    EndIf
                EndIf
            Next
        EndIf
        
        If selectedToken = TK_END Then Exit
        
        ' 4.3 Actualizar buffer de tokens
        For Local shiftIdx:Int = recentTokens.Length-1 To 1 Step -1
            recentTokens[shiftIdx] = recentTokens[shiftIdx-1]
        Next
        recentTokens[0] = selectedToken
        
        ' 4.4 Construir respuesta
        Local newToken:String = tokenDB[selectedToken]
        If generatedText.Length > 0 And Not IsPunctuation(newToken) And Not newToken.StartsWith(" ")
            generatedText :+ " "
        EndIf
        generatedText :+ newToken
        currentToken = selectedToken
        generatedCount :+ 1
    Wend
    
    ' 5. Finalizaci髇
    generatedText = generatedText.Trim()
    AddToHistory(generatedText, False)
    Return generatedText
End Method
    
    Method IsPunctuation:Int(token:String)
        Return token = "." Or token = "," Or token = "!" Or token = "?" Or token = ";" Or token = ":"
    End Method
    
    Method IsLower:Int(c:String)
        Return (c >= Asc("a") And c <= Asc("z")) Or (c >= Asc("谩") And c <= Asc("煤"))
    End Method
    
    Method ContarTokensActivos:Int()
        Local count_tokact:Int = 0
        For tokenID_tokact:Int = 0 To MAX_TOKENS-1
            If tokenDB[tokenID_tokact] And tokenDB[tokenID_tokact] <> "" Then count_tokact :+ 1
        Next
        Return count_tokact
    End Method
    
    Method GuardarModelo(archivo:String, fullSave:Int = True)
        Local stream_save:TStream = WriteStream(archivo)
        If Not stream_save Then Return
        
        ' Encabezado
        stream_save.WriteLine("BLLMv6")
        stream_save.WriteLine(String(learningEnabled))
        stream_save.WriteLine(String(totalInteractions))
        stream_save.WriteLine(enginePrompt)
        
        ' Tokens
        stream_save.WriteLine("TOKENS")
        Local numTokens_save:Int = 0
        For tokenID_save:Int = 0 To MAX_TOKENS-1
            If tokenDB[tokenID_save] And tokenDB[tokenID_save] <> "" Then numTokens_save :+ 1
        Next
        stream_save.WriteLine(numTokens_save)
        
        For tokenID_save2:Int = 0 To MAX_TOKENS-1
            If tokenDB[tokenID_save2] And tokenDB[tokenID_save2] <> ""
                stream_save.WriteLine(tokenID_save2 + "|" + tokenDB[tokenID_save2] + "|" + tokenCounts[tokenID_save2])
            EndIf
        Next
        
        ' Embeddings (solo tokens usados)
        stream_save.WriteLine("EMBEDDINGS")
        For tokenID_emb:Int = 0 To MAX_TOKENS-1
            If tokenDB[tokenID_emb] And tokenDB[tokenID_emb] <> ""
                Local linea_emb:String = tokenID_emb + "|"
                For embIdx_emb:Int = 0 To EMBEDDING_SIZE-1
                    linea_emb :+ String(embeddingWeights[tokenID_emb, embIdx_emb]) + ","
                Next
                stream_save.WriteLine(linea_emb)
            EndIf
        Next
        
        ' Atenci贸n (completa)
        stream_save.WriteLine("ATTENTION")
        For headID_att:Int = 0 To 7
            For rowIdx_att:Int = 0 To EMBEDDING_SIZE-1
                Local linea_att:String = headID_att + "|" + rowIdx_att + "|"
                For colIdx_att:Int = 0 To EMBEDDING_SIZE-1
                    linea_att :+ String(attentionWeights[headID_att, rowIdx_att, colIdx_att]) + ","
                Next
                stream_save.WriteLine(linea_att)
            Next
        Next
        
        ' FFN weights
        stream_save.WriteLine("FFN_LAYER1")
        For i_ffn1:Int = 0 Until EMBEDDING_SIZE
            Local linea_ffn1:String = i_ffn1 + "|"
            For j_ffn1:Int = 0 Until FFN_HIDDEN_SIZE
                linea_ffn1 :+ String(ffn.weights1[i_ffn1, j_ffn1]) + ","
            Next
            stream_save.WriteLine(linea_ffn1)
        Next
        
        stream_save.WriteLine("FFN_LAYER2")
        For i_ffn2:Int = 0 Until FFN_HIDDEN_SIZE
            Local linea_ffn2:String = i_ffn2 + "|"
            For j_ffn2:Int = 0 Until EMBEDDING_SIZE
                linea_ffn2 :+ String(ffn.weights2[i_ffn2, j_ffn2]) + ","
            Next
            stream_save.WriteLine(linea_ffn2)
        Next
        
        stream_save.WriteLine("FFN_BIASES")
        Local linea_bias1:String = "1|"
        For b_ffn1:Int = 0 Until FFN_HIDDEN_SIZE
            linea_bias1 :+ String(ffn.bias1[b_ffn1]) + ","
        Next
        stream_save.WriteLine(linea_bias1)
        
        Local linea_bias2:String = "2|"
        For b_ffn2:Int = 0 Until EMBEDDING_SIZE
            linea_bias2 :+ String(ffn.bias2[b_ffn2]) + ","
        Next
        stream_save.WriteLine(linea_bias2)
        
        ' Memoria e historial si es fullSave
        If fullSave
            stream_save.WriteLine("MEMORY")
            Local mem_save:Float[] = memory.GetContext()
            For emb_memsave:Int = 0 To EMBEDDING_SIZE-1
                stream_save.WriteFloat(mem_save[emb_memsave])
            Next
            
            stream_save.WriteLine("HISTORY")
            stream_save.WriteLine(conversationHistory.Length)
            For msg_save:String = EachIn conversationHistory
                stream_save.WriteLine(msg_save)
            Next
        EndIf
        
        stream_save.Close()
    End Method
    
    Method CargarModelo:Int(archivo:String, loadFull:Int = False)
    Local stream_load:TStream = ReadStream(archivo)
    If Not stream_load Then Return False
    
    ' Verificar versi贸n del archivo
    Local version:String = stream_load.ReadLine()
    If version <> "BLLMv6"
        stream_load.Close()
        Print "Error: Versi贸n de archivo no compatible. Esperaba BLLMv6, obtuvo " + version
        Return False
    EndIf
    
    ' Configuraci贸n b谩sica
    learningEnabled = Int(stream_load.ReadLine())
    totalInteractions = Long(stream_load.ReadLine())
    enginePrompt = stream_load.ReadLine()
    
    ' Cargar secciones
    While Not stream_load.EOF()
        Local section_load:String = stream_load.ReadLine()
        
        Select section_load
            Case "TOKENS"
                Local numTokens_load:Int = Int(stream_load.ReadLine())
                For Local n_load:Int = 1 To numTokens_load
                    Local linea_load:String = stream_load.ReadLine()
                    If linea_load = "" Then Continue
                    
                    Local partes_load:String[] = linea_load.Split("|")
                    If partes_load.Length < 3 Then Continue
                    
                    Local id_load:Int = Int(partes_load[0])
                    Local token_str:String = partes_load[1]
                    Local count_str:String = partes_load[2]
                    
                    ' Validar ID y asignar token
                    If id_load >= 0 And id_load < MAX_TOKENS
                        tokenDB[id_load] = token_str
                        
                        ' Asignar conteo con validaci贸n
                        If count_str <> ""
                            tokenCounts[id_load] = Int(count_str)
                        Else
                            tokenCounts[id_load] = 1 ' Valor por defecto
                        EndIf
                    Else
                        Print "Advertencia: ID de token inv谩lido " + id_load + " - Token: " + token_str
                    EndIf
                Next
                
            Case "EMBEDDINGS"
                While Not stream_load.EOF()
                    Local linea_embload:String = stream_load.ReadLine()
                    If linea_embload = "" Or linea_embload = "ATTENTION" Or linea_embload = "FFN_LAYER1" Then Exit
                    
                    Local partes_embload:String[] = linea_embload.Split("|")
                    If partes_embload.Length < 2 Then Continue
                    
                    Local id_embload:Int = Int(partes_embload[0])
                    Local valores_embload:String[] = partes_embload[1].Split(",")
                    
                    If id_embload >= 0 And id_embload < MAX_TOKENS
                        For Local embIdx_embload:Int = 0 Until Min(valores_embload.Length, EMBEDDING_SIZE)
                            If valores_embload[embIdx_embload] <> ""
                                embeddingWeights[id_embload, embIdx_embload] = Float(valores_embload[embIdx_embload])
                            EndIf
                        Next
                    EndIf
                Wend
                
            Case "ATTENTION"
                While Not stream_load.EOF()
                    Local linea_attload:String = stream_load.ReadLine()
                    If linea_attload = "" Or linea_attload = "FFN_LAYER1" Or linea_attload = "MEMORY" Or linea_attload = "HISTORY" Then Exit
                    
                    Local partes_attload:String[] = linea_attload.Split("|")
                    If partes_attload.Length < 3 Then Continue
                    
                    Local h_attload:Int = Int(partes_attload[0])
                    Local i_attload:Int = Int(partes_attload[1])
                    Local valores_attload:String[] = partes_attload[2].Split(",")
                    
                    If h_attload >= 0 And h_attload < 8 And i_attload >= 0 And i_attload < EMBEDDING_SIZE
                        For Local j_attload:Int = 0 Until Min(valores_attload.Length, EMBEDDING_SIZE)
                            If valores_attload[j_attload] <> ""
                                attentionWeights[h_attload, i_attload, j_attload] = Float(valores_attload[j_attload])
                            EndIf
                        Next
                    EndIf
                Wend
                
            Case "FFN_LAYER1"
                While Not stream_load.EOF()
                    Local linea_ffn1load:String = stream_load.ReadLine()
                    If linea_ffn1load = "" Or linea_ffn1load = "FFN_LAYER2" Then Exit
                    
                    Local partes_ffn1load:String[] = linea_ffn1load.Split("|")
                    If partes_ffn1load.Length < 2 Then Continue
                    
                    Local i_ffn1load:Int = Int(partes_ffn1load[0])
                    Local valores_ffn1load:String[] = partes_ffn1load[1].Split(",")
                    
                    If i_ffn1load >= 0 And i_ffn1load < EMBEDDING_SIZE
                        For Local j_ffn1load:Int = 0 Until Min(valores_ffn1load.Length, FFN_HIDDEN_SIZE)
                            If valores_ffn1load[j_ffn1load] <> ""
                                ffn.weights1[i_ffn1load, j_ffn1load] = Float(valores_ffn1load[j_ffn1load])
                            EndIf
                        Next
                    EndIf
                Wend
                
            Case "FFN_LAYER2"
                While Not stream_load.EOF()
                    Local linea_ffn2load:String = stream_load.ReadLine()
                    If linea_ffn2load = "" Or linea_ffn2load = "FFN_BIASES" Then Exit
                    
                    Local partes_ffn2load:String[] = linea_ffn2load.Split("|")
                    If partes_ffn2load.Length < 2 Then Continue
                    
                    Local i_ffn2load:Int = Int(partes_ffn2load[0])
                    Local valores_ffn2load:String[] = partes_ffn2load[1].Split(",")
                    
                    If i_ffn2load >= 0 And i_ffn2load < FFN_HIDDEN_SIZE
                        For Local j_ffn2load:Int = 0 Until Min(valores_ffn2load.Length, EMBEDDING_SIZE)
                            If valores_ffn2load[j_ffn2load] <> ""
                                ffn.weights2[i_ffn2load, j_ffn2load] = Float(valores_ffn2load[j_ffn2load])
                            EndIf
                        Next
                    EndIf
                Wend
                
            Case "FFN_BIASES"
                Local linea_biasload:String = stream_load.ReadLine()
                If linea_biasload <> ""
                    Local partes_biasload:String[] = linea_biasload.Split("|")
                    If partes_biasload.Length >= 2
                        Local layer_biasload:Int = Int(partes_biasload[0])
                        Local valores_biasload:String[] = partes_biasload[1].Split(",")
                        
                        If layer_biasload = 1
                            For Local b_ffn1load:Int = 0 Until Min(valores_biasload.Length, FFN_HIDDEN_SIZE)
                                If valores_biasload[b_ffn1load] <> ""
                                    ffn.bias1[b_ffn1load] = Float(valores_biasload[b_ffn1load])
                                EndIf
                            Next
                        ElseIf layer_biasload = 2
                            For Local b_ffn2load:Int = 0 Until Min(valores_biasload.Length, EMBEDDING_SIZE)
                                If valores_biasload[b_ffn2load] <> ""
                                    ffn.bias2[b_ffn2load] = Float(valores_biasload[b_ffn2load])
                                EndIf
                            Next
                        EndIf
                    EndIf
                EndIf
                
            Case "MEMORY"
                If loadFull
                    For Local emb_memload:Int = 0 Until EMBEDDING_SIZE
                        memory.GetContext()[emb_memload] = stream_load.ReadFloat()
                    Next
                EndIf
                
            Case "HISTORY"
                If loadFull
                    Local historySize_load:Int = Int(stream_load.ReadLine())
                    conversationHistory = New String[historySize_load]
                    For Local histIdx_load:Int = 0 Until historySize_load
                        conversationHistory[histIdx_load] = stream_load.ReadLine()
                    Next
                EndIf
        End Select
    Wend
    
    stream_load.Close()
    
    ' Reconstruir contexto si es necesario
    If conversationHistory.Length > 0 And loadFull
        Local fullContext_load:String = ""
        For Local msg_load:String = EachIn conversationHistory
            fullContext_load :+ msg_load + "~n"
        Next
        
        Local inputTokens_load:String[] = Tokenizar(fullContext_load)
        Local inputIDs_load:Int[] = New Int[inputTokens_load.Length]
        
        For Local tokenPos_load:Int = 0 Until inputTokens_load.Length
            inputIDs_load[tokenPos_load] = ObtenerOCrearID(inputTokens_load[tokenPos_load])
        Next
        
        ProcesarEntrada(inputIDs_load)
    EndIf
    
    Return True
End Method
End Type