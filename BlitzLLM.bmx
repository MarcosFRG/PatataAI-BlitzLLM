' Constantes del sistema
Const MAX_TOKENS:Int = 1024 ' 50257
Const EMBEDDING_SIZE:Int = 512 ' 1024
Const MAX_CONTEXT:Int = 256 ' 2048
Const LEARNING_RATE:Float = 0.01 ' 0.0001
Const MAX_RESPONSE_LENGTH:Int = 100 ' 500
Const IS_LEARNING:Int = 1
Const TEMPERATURE:Float = 0.7
Const MEMORY_DECAY:Float = 0.95
Const BATCH_SIZE:Int = 32
Const CHECKPOINT_EVERY:Int = 1000
Const FFN_HIDDEN_SIZE:Int = 2048 ' 4096
Const FFN_DROPOUT:Float = 0.1
Const FFN_ACTIVATION_SCALE:Float = 0.1
Const NUM_ATTENTION_HEADS:Int = 16
Const NUM_LAYERS:Int = 12
Const TK_UNKNOWN:Int = 0
Const TK_START:Int = 1
Const TK_END:Int = 2
Const TK_RES:Int = 3
Const TK_PROMPT:Int = 4
Const TK_NEWLINE:Int = 5
Const TK_PAD:Int = 6
' Configuración de contexto/historial
Const USAR_HISTORIAL:Int = False ' True para activar memoria de conversación
Const MAX_HISTORIAL:Int = 3      ' Máximo mensajes a recordar (si USAR_HISTORIAL=True)
Const LIMPIAR_CONTEXTO_ENTRE_PROMTPS:Int = True ' Para pruebas iniciales
' Constantes de logging
Const LOG_ACTIVADO:Int = False ' Cambiar a False para desactivar logs
Const LOG_TO_FILE:Int = True ' Guardar logs en archivo
Const LOG_FILE:String = "llm_debug.log" ' Nombre del archivo de log
Const LOG_TOKENIZACION:Int = True
Const LOG_GENERACION:Int = True
Const LOG_ENTRENAMIENTO:Int = True
Const LOG_PENALIZACIONES:Int = False ' Solo activar para depuración avanzada

Type TBpePair
    Field left:Int
    Field right:Int
    Field count:Int
    Field score:Float
End Type

Type THierarchicalMemory
    Field shortTerm:Float[]
    Field mediumTerm:Float[]
    Field longTerm:Float[]
    Field updateCounter:Int
    Field importanceScores:Float[]
    
    Method New()
        shortTerm = New Float[EMBEDDING_SIZE]
        mediumTerm = New Float[EMBEDDING_SIZE]
        longTerm = New Float[EMBEDDING_SIZE]
        importanceScores = New Float[EMBEDDING_SIZE]
        Clear()
    End Method
    
    Method UpdateMemory(contextData:Float[], importance:Float = 1.0)
        Local normVal:Float = 0.0
        For embPos:Int = 0 Until EMBEDDING_SIZE
            normVal :+ contextData[embPos] * contextData[embPos]
        Next
        normVal = Sqr(normVal)
        
        For embPos:Int = 0 Until EMBEDDING_SIZE
            importanceScores[embPos] = 0.9 * importanceScores[embPos] + 0.1 * (contextData[embPos] * contextData[embPos]) / (normVal * normVal + 0.0001)
            
            Local alphaVal:Float = 0.1 * (1.0 + importanceScores[embPos]) * importance
            shortTerm[embPos] = (1.0 - alphaVal) * shortTerm[embPos] + alphaVal * contextData[embPos]
            
            If updateCounter Mod 5 = 0
                mediumTerm[embPos] = 0.95 * mediumTerm[embPos] + 0.05 * shortTerm[embPos]
            EndIf
            
            If updateCounter Mod 50 = 0
                longTerm[embPos] = 0.99 * longTerm[embPos] + 0.01 * mediumTerm[embPos]
            EndIf
        Next
        
        updateCounter :+ 1
    End Method
    
    Method GetContext:Float[]()
        Local resultArr:Float[] = New Float[EMBEDDING_SIZE]
        For embPos:Int = 0 Until EMBEDDING_SIZE
            Local shortWeight:Float = 0.6 * (1.0 + importanceScores[embPos])
            Local mediumWeight:Float = 0.3
            Local longWeight:Float = 0.1 / (1.0 + importanceScores[embPos])
            
            Local totalWeight:Float = shortWeight + mediumWeight + longWeight
            resultArr[embPos] = (shortWeight * shortTerm[embPos] + mediumWeight * mediumTerm[embPos] + longWeight * longTerm[embPos]) / totalWeight
        Next
        Return resultArr
    End Method
    
    Method Clear()
        For embPos:Int = 0 Until EMBEDDING_SIZE
            shortTerm[embPos] = 0
            mediumTerm[embPos] = 0
            longTerm[embPos] = 0
            importanceScores[embPos] = 0.5
        Next
        updateCounter = 0
    End Method
End Type

Type TFeedForwardNetwork
    Field weights1:Float[]
    Field weights2:Float[]
    Field bias1:Float[]
    Field bias2:Float[]
    
    Method New()
        weights1 = New Float[EMBEDDING_SIZE * FFN_HIDDEN_SIZE]
        weights2 = New Float[FFN_HIDDEN_SIZE * EMBEDDING_SIZE]
        bias1 = New Float[FFN_HIDDEN_SIZE]
        bias2 = New Float[EMBEDDING_SIZE]
        InitializeWeights()
    End Method
    
    Method InitializeWeights()
        Local scaleVal1:Float = Sqr(2.0 / EMBEDDING_SIZE)
        For weightPos1:Int = 0 Until EMBEDDING_SIZE
            For weightPos2:Int = 0 Until FFN_HIDDEN_SIZE
                weights1[weightPos1 * FFN_HIDDEN_SIZE + weightPos2] = Rnd(-scaleVal1, scaleVal1)
            Next
        Next
        
        Local scaleVal2:Float = Sqr(2.0 / FFN_HIDDEN_SIZE)
        For weightPos3:Int = 0 Until FFN_HIDDEN_SIZE
            For weightPos4:Int = 0 Until EMBEDDING_SIZE
                weights2[weightPos3 * EMBEDDING_SIZE + weightPos4] = Rnd(-scaleVal2, scaleVal2)
            Next
        Next
        
        For biasPos1:Int = 0 Until FFN_HIDDEN_SIZE
            bias1[biasPos1] = 0.0
        Next
        
        For biasPos2:Int = 0 Until EMBEDDING_SIZE
            bias2[biasPos2] = 0.0
        Next
    End Method
    
    Method Process:Float[](inputData:Float[])
        Local hiddenLayer:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For hiddenPos:Int = 0 Until FFN_HIDDEN_SIZE
            hiddenLayer[hiddenPos] = bias1[hiddenPos]
            
            For inputPos:Int = 0 Until EMBEDDING_SIZE
                hiddenLayer[hiddenPos] :+ inputData[inputPos] * weights1[inputPos * FFN_HIDDEN_SIZE + hiddenPos]
            Next
            
            hiddenLayer[hiddenPos] = 0.5 * hiddenLayer[hiddenPos] * (1.0 + Tanh(Sqr(2.0 / Pi) * (hiddenLayer[hiddenPos] + 0.044715 * hiddenLayer[hiddenPos] * hiddenLayer[hiddenPos] * hiddenLayer[hiddenPos])))
            
            If IS_LEARNING And Rnd() < FFN_DROPOUT
                hiddenLayer[hiddenPos] = 0
            EndIf
        Next
        
        Local outputLayer:Float[] = New Float[EMBEDDING_SIZE]
        
        For outputPos:Int = 0 Until EMBEDDING_SIZE
            outputLayer[outputPos] = bias2[outputPos]
            
            For hiddenPos2:Int = 0 Until FFN_HIDDEN_SIZE
                outputLayer[outputPos] :+ hiddenLayer[hiddenPos2] * weights2[hiddenPos2 * EMBEDDING_SIZE + outputPos]
            Next
        Next
        
        Return outputLayer
    End Method
    
    Method Backpropagate(inputData:Float[], gradOutputData:Float[], gradInputData:Float[] Var, gradWeights1Data:Float[] Var, gradWeights2Data:Float[] Var, gradBias1Data:Float[] Var, gradBias2Data:Float[] Var)
        Local hiddenLayer2:Float[] = New Float[FFN_HIDDEN_SIZE]
        Local hiddenPreAct:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For hiddenPos3:Int = 0 Until FFN_HIDDEN_SIZE
            hiddenPreAct[hiddenPos3] = bias1[hiddenPos3]
            
            For inputPos2:Int = 0 Until EMBEDDING_SIZE
                hiddenPreAct[hiddenPos3] :+ inputData[inputPos2] * weights1[inputPos2 * FFN_HIDDEN_SIZE + hiddenPos3]
            Next
            
            hiddenLayer2[hiddenPos3] = 0.5 * hiddenPreAct[hiddenPos3] * (1.0 + Tanh(Sqr(2.0 / Pi) * (hiddenPreAct[hiddenPos3] + 0.044715 * hiddenPreAct[hiddenPos3] * hiddenPreAct[hiddenPos3] * hiddenPreAct[hiddenPos3])))
        Next
        
        For outputPos2:Int = 0 Until EMBEDDING_SIZE
            If gradBias2Data <> Null Then gradBias2Data[outputPos2] :+ gradOutputData[outputPos2] * LEARNING_RATE
            
            If gradWeights2Data <> Null
                For hiddenPos4:Int = 0 Until FFN_HIDDEN_SIZE
                    gradWeights2Data[hiddenPos4 * EMBEDDING_SIZE + outputPos2] :+ hiddenLayer2[hiddenPos4] * gradOutputData[outputPos2] * LEARNING_RATE
                Next
            EndIf
        Next
        
        Local gradHiddenLayer:Float[] = New Float[FFN_HIDDEN_SIZE]
        
        For hiddenPos5:Int = 0 Until FFN_HIDDEN_SIZE
            Local sumVal:Float = 0
            For outputPos3:Int = 0 Until EMBEDDING_SIZE
                sumVal :+ gradOutputData[outputPos3] * weights2[hiddenPos5 * EMBEDDING_SIZE + outputPos3]
            Next
            
            Local xVal:Float = hiddenPreAct[hiddenPos5]
            Local geluDerivVal:Float = 0.5 * Tanh(0.0356774 * xVal * xVal * xVal + 0.797885 * xVal) + (0.0535161 * xVal * xVal * xVal + 0.398942 * xVal) / (Cosh(0.0356774 * xVal * xVal * xVal + 0.797885 * xVal) ^ 2) + 0.5
            
            gradHiddenLayer[hiddenPos5] = sumVal * geluDerivVal
        Next
        
        For hiddenPos6:Int = 0 Until FFN_HIDDEN_SIZE
            If gradBias1Data <> Null Then gradBias1Data[hiddenPos6] :+ gradHiddenLayer[hiddenPos6] * LEARNING_RATE
            
            If gradWeights1Data <> Null
                For inputPos3:Int = 0 Until EMBEDDING_SIZE
                    gradWeights1Data[inputPos3 * FFN_HIDDEN_SIZE + hiddenPos6] :+ inputData[inputPos3] * gradHiddenLayer[hiddenPos6] * LEARNING_RATE
                Next
            EndIf
        Next
        
        If gradInputData <> Null
            For inputPos4:Int = 0 Until EMBEDDING_SIZE
                gradInputData[inputPos4] = 0
                
                For hiddenPos7:Int = 0 Until FFN_HIDDEN_SIZE
                    gradInputData[inputPos4] :+ gradHiddenLayer[hiddenPos7] * weights1[inputPos4 * FFN_HIDDEN_SIZE + hiddenPos7]
                Next
            Next
        EndIf
    End Method
End Type

Type TAttentionLayer
    Field qWeights:Float[]
    Field kWeights:Float[]
    Field vWeights:Float[]
    Field oWeights:Float[]
    
    Method New()
        qWeights = New Float[EMBEDDING_SIZE * EMBEDDING_SIZE]
        kWeights = New Float[EMBEDDING_SIZE * EMBEDDING_SIZE]
        vWeights = New Float[EMBEDDING_SIZE * EMBEDDING_SIZE]
        oWeights = New Float[EMBEDDING_SIZE * EMBEDDING_SIZE]
        InitializeWeights()
    End Method
    
    Method InitializeWeights()
        Local scaleVal3:Float = Sqr(2.0 / (EMBEDDING_SIZE + EMBEDDING_SIZE))
        
        For weightPos5:Int = 0 Until EMBEDDING_SIZE
            For weightPos6:Int = 0 Until EMBEDDING_SIZE
                Local idxVal:Int = weightPos5 * EMBEDDING_SIZE + weightPos6
                qWeights[idxVal] = Rnd(-scaleVal3, scaleVal3)
                kWeights[idxVal] = Rnd(-scaleVal3, scaleVal3)
                vWeights[idxVal] = Rnd(-scaleVal3, scaleVal3)
                oWeights[idxVal] = Rnd(-scaleVal3, scaleVal3)
            Next
        Next
    End Method
    
    Method Apply:Float[](inputData2:Float[], maskData:Float[] = Null)
        Local qVals:Float[] = New Float[EMBEDDING_SIZE]
        Local kVals:Float[] = New Float[EMBEDDING_SIZE]
        Local vVals:Float[] = New Float[EMBEDDING_SIZE]
        
        For attnPos1:Int = 0 Until EMBEDDING_SIZE
            For attnPos2:Int = 0 Until EMBEDDING_SIZE
                qVals[attnPos2] :+ inputData2[attnPos1] * qWeights[attnPos1 * EMBEDDING_SIZE + attnPos2]
                kVals[attnPos2] :+ inputData2[attnPos1] * kWeights[attnPos1 * EMBEDDING_SIZE + attnPos2]
                vVals[attnPos2] :+ inputData2[attnPos1] * vWeights[attnPos1 * EMBEDDING_SIZE + attnPos2]
            Next
        Next
        
        Local scoreVals:Float[] = New Float[EMBEDDING_SIZE]
        Local sumExpVal:Float = 0.0
        
        For attnPos3:Int = 0 Until EMBEDDING_SIZE
            scoreVals[attnPos3] = 0.0
            For attnPos4:Int = 0 Until EMBEDDING_SIZE
                scoreVals[attnPos3] :+ qVals[attnPos4] * kVals[attnPos4] / Sqr(EMBEDDING_SIZE)
            Next
            
            If maskData <> Null And maskData.Length > attnPos3
                scoreVals[attnPos3] :+ maskData[attnPos3]
            EndIf
            
            scoreVals[attnPos3] = Exp(scoreVals[attnPos3])
            sumExpVal :+ scoreVals[attnPos3]
        Next
        
        Local outputVals:Float[] = New Float[EMBEDDING_SIZE]
        If sumExpVal > 0.0
            For attnPos5:Int = 0 Until EMBEDDING_SIZE
                scoreVals[attnPos5] :/ sumExpVal
                For attnPos6:Int = 0 Until EMBEDDING_SIZE
                    outputVals[attnPos5] :+ scoreVals[attnPos6] * vVals[attnPos6]
                Next
            Next
        EndIf
        
        Local finalOutputVals:Float[] = New Float[EMBEDDING_SIZE]
        For attnPos7:Int = 0 Until EMBEDDING_SIZE
            For attnPos8:Int = 0 Until EMBEDDING_SIZE
                finalOutputVals[attnPos8] :+ outputVals[attnPos7] * oWeights[attnPos7 * EMBEDDING_SIZE + attnPos8]
            Next
        Next
        
        Return finalOutputVals
    End Method
    
    Method Backpropagate(inputData3:Float[], gradOutputData2:Float[], gradInputData2:Float[] Var, gradQData:Float[] Var, gradKData:Float[] Var, gradVData:Float[] Var, gradOData:Float[] Var)
        Local qVals2:Float[] = New Float[EMBEDDING_SIZE]
        Local kVals2:Float[] = New Float[EMBEDDING_SIZE]
        Local vVals2:Float[] = New Float[EMBEDDING_SIZE]
        
        For attnPos9:Int = 0 Until EMBEDDING_SIZE
            For attnPos10:Int = 0 Until EMBEDDING_SIZE
                qVals2[attnPos10] :+ inputData3[attnPos9] * qWeights[attnPos9 * EMBEDDING_SIZE + attnPos10]
                kVals2[attnPos10] :+ inputData3[attnPos9] * kWeights[attnPos9 * EMBEDDING_SIZE + attnPos10]
                vVals2[attnPos10] :+ inputData3[attnPos9] * vWeights[attnPos9 * EMBEDDING_SIZE + attnPos10]
            Next
        Next
        
        Local scoreVals2:Float[] = New Float[EMBEDDING_SIZE]
        Local expScoreVals:Float[] = New Float[EMBEDDING_SIZE]
        Local sumExpVal2:Float = 0.0
        
        For attnPos11:Int = 0 Until EMBEDDING_SIZE
            scoreVals2[attnPos11] = 0.0
            For attnPos12:Int = 0 Until EMBEDDING_SIZE
                scoreVals2[attnPos11] :+ qVals2[attnPos12] * kVals2[attnPos12] / Sqr(EMBEDDING_SIZE)
            Next
            
            expScoreVals[attnPos11] = Exp(scoreVals2[attnPos11])
            sumExpVal2 :+ expScoreVals[attnPos11]
        Next
        
        Local attentionVals:Float[] = New Float[EMBEDDING_SIZE]
        If sumExpVal2 > 0.0
            For attnPos13:Int = 0 Until EMBEDDING_SIZE
                expScoreVals[attnPos13] :/ sumExpVal2
                For attnPos14:Int = 0 Until EMBEDDING_SIZE
                    attentionVals[attnPos13] :+ expScoreVals[attnPos14] * vVals2[attnPos14]
                Next
            Next
        EndIf
        
        Local gradAttentionVals:Float[] = New Float[EMBEDDING_SIZE]
        For attnPos15:Int = 0 Until EMBEDDING_SIZE
            For attnPos16:Int = 0 Until EMBEDDING_SIZE
                gradAttentionVals[attnPos15] :+ gradOutputData2[attnPos16] * oWeights[attnPos15 * EMBEDDING_SIZE + attnPos16]
            Next
        Next
        
        Local gradVVals:Float[] = New Float[EMBEDDING_SIZE]
        Local gradExpScoreVals:Float[] = New Float[EMBEDDING_SIZE]
        
        For attnPos17:Int = 0 Until EMBEDDING_SIZE
            For attnPos18:Int = 0 Until EMBEDDING_SIZE
                gradVVals[attnPos17] :+ gradAttentionVals[attnPos18] * expScoreVals[attnPos18]
                gradExpScoreVals[attnPos18] :+ gradAttentionVals[attnPos18] * vVals2[attnPos17]
            Next
        Next
        
        Local gradScoreVals:Float[] = New Float[EMBEDDING_SIZE]
        For attnPos19:Int = 0 Until EMBEDDING_SIZE
            gradScoreVals[attnPos19] = gradExpScoreVals[attnPos19] * expScoreVals[attnPos19] * (1.0 - expScoreVals[attnPos19] / sumExpVal2)
        Next
        
        Local gradQVals:Float[] = New Float[EMBEDDING_SIZE]
        Local gradKVals:Float[] = New Float[EMBEDDING_SIZE]
        
        For attnPos20:Int = 0 Until EMBEDDING_SIZE
            For attnPos21:Int = 0 Until EMBEDDING_SIZE
                gradQVals[attnPos20] :+ gradScoreVals[attnPos21] * kVals2[attnPos21] / Sqr(EMBEDDING_SIZE)
                gradKVals[attnPos20] :+ gradScoreVals[attnPos21] * qVals2[attnPos21] / Sqr(EMBEDDING_SIZE)
            Next
        Next
        
        If gradQData <> Null And gradKData <> Null And gradVData <> Null And gradOData <> Null
            For attnPos22:Int = 0 Until EMBEDDING_SIZE
                For attnPos23:Int = 0 Until EMBEDDING_SIZE
                    gradQData[attnPos22 * EMBEDDING_SIZE + attnPos23] :+ gradQVals[attnPos23] * inputData3[attnPos22] * LEARNING_RATE
                    gradKData[attnPos22 * EMBEDDING_SIZE + attnPos23] :+ gradKVals[attnPos23] * inputData3[attnPos22] * LEARNING_RATE
                    gradVData[attnPos22 * EMBEDDING_SIZE + attnPos23] :+ gradVVals[attnPos23] * inputData3[attnPos22] * LEARNING_RATE
                    gradOData[attnPos22 * EMBEDDING_SIZE + attnPos23] :+ gradOutputData2[attnPos23] * attentionVals[attnPos22] * LEARNING_RATE
                Next
            Next
        EndIf
        
        If gradInputData2 <> Null
            For attnPos24:Int = 0 Until EMBEDDING_SIZE
                gradInputData2[attnPos24] = 0
                For attnPos25:Int = 0 Until EMBEDDING_SIZE
                    gradInputData2[attnPos24] :+ gradQVals[attnPos25] * qWeights[attnPos24 * EMBEDDING_SIZE + attnPos25]
                    gradInputData2[attnPos24] :+ gradKVals[attnPos25] * kWeights[attnPos24 * EMBEDDING_SIZE + attnPos25]
                    gradInputData2[attnPos24] :+ gradVVals[attnPos25] * vWeights[attnPos24 * EMBEDDING_SIZE + attnPos25]
                Next
            Next
        EndIf
    End Method
End Type

Type TTransformerBlock
    Field attention:TAttentionLayer
    Field ffn:TFeedForwardNetwork
    Field norm1:Float[]
    Field norm2:Float[]
    
    Method New()
        attention = New TAttentionLayer
        ffn = New TFeedForwardNetwork
        norm1 = New Float[EMBEDDING_SIZE]
        norm2 = New Float[EMBEDDING_SIZE]
        
        For normPos1:Int = 0 Until EMBEDDING_SIZE
            norm1[normPos1] = 1.0
            norm2[normPos1] = 1.0
        Next
    End Method
    
    Method Process:Float[](inputData4:Float[], maskData2:Float[] = Null)
        Local normInput1:Float[] = LayerNorm(inputData4, norm1)
        Local attnOut:Float[] = attention.Apply(normInput1, maskData2)
        
        For resPos1:Int = 0 Until EMBEDDING_SIZE
            attnOut[resPos1] :+ inputData4[resPos1]
        Next
        
        Local normInput2:Float[] = LayerNorm(attnOut, norm2)
        Local ffnOut:Float[] = ffn.Process(normInput2)
        
        For resPos2:Int = 0 Until EMBEDDING_SIZE
            ffnOut[resPos2] :+ attnOut[resPos2]
        Next
        
        Return ffnOut
    End Method
    
    Method LayerNorm:Float[](xData:Float[], gammaData:Float[])
        Local meanVal:Float = 0.0
        Local varianceVal:Float = 0.0
        
        For normPos2:Int = 0 Until EMBEDDING_SIZE
            meanVal :+ xData[normPos2]
        Next
        meanVal :/ EMBEDDING_SIZE
        
        For normPos3:Int = 0 Until EMBEDDING_SIZE
            varianceVal :+ (xData[normPos3] - meanVal) * (xData[normPos3] - meanVal)
        Next
        varianceVal :/ EMBEDDING_SIZE
        
        Local resultData:Float[] = New Float[EMBEDDING_SIZE]
        For normPos4:Int = 0 Until EMBEDDING_SIZE
            resultData[normPos4] = gammaData[normPos4] * (xData[normPos4] - meanVal) / Sqr(varianceVal + 0.0001)
        Next
        
        Return resultData
    End Method
    
Method Backpropagate(inputData5:Float[], gradOutputData3:Float[], gradInputData3:Float[] Var, gradNorm1Data:Float[] Var, gradNorm2Data:Float[] Var)
    Local normInput1Data:Float[] = LayerNorm(inputData5, norm1)
    Local attnOutData:Float[] = attention.Apply(normInput1Data)
    
    Local normInput2Data:Float[] = LayerNorm(attnOutData, norm2)
    Local ffnOutData:Float[] = ffn.Process(normInput2Data)
    
    Local gradFfnOut:Float[] = New Float[EMBEDDING_SIZE]
    Local gradAttnOut:Float[] = New Float[EMBEDDING_SIZE]
    
    For gradPos1:Int = 0 Until EMBEDDING_SIZE
        gradFfnOut[gradPos1] = gradOutputData3[gradPos1]
        gradAttnOut[gradPos1] = gradOutputData3[gradPos1]
    Next

    Local dummyGradWeights1:Float[]
    Local dummyGradWeights2:Float[]
    Local dummyGradBias1:Float[]
    Local dummyGradBias2:Float[]
    
    Local gradNormInput2:Float[] = New Float[EMBEDDING_SIZE]
    ffn.Backpropagate(normInput2Data, gradFfnOut, gradNormInput2, dummyGradWeights1, dummyGradWeights2, dummyGradBias1, dummyGradBias2)
    
    For gradPos2:Int = 0 Until EMBEDDING_SIZE
        gradAttnOut[gradPos2] :+ gradNormInput2[gradPos2]
    Next

    Local dummyGradQ:Float[]
    Local dummyGradK:Float[]
    Local dummyGradV:Float[]
    Local dummyGradO:Float[]
    
    Local gradNormInput1:Float[] = New Float[EMBEDDING_SIZE]
    attention.Backpropagate(normInput1Data, gradAttnOut, gradNormInput1, dummyGradQ, dummyGradK, dummyGradV, dummyGradO)
    
    For gradPos3:Int = 0 Until EMBEDDING_SIZE
        If gradInputData3 <> Null Then gradInputData3[gradPos3] = gradNormInput1[gradPos3] + gradAttnOut[gradPos3]
    Next
    
    Local gradX1:Float[] = New Float[EMBEDDING_SIZE]
    Local gradX2:Float[] = New Float[EMBEDDING_SIZE]
    
    Local meanVal2:Float = 0.0
    Local varianceVal2:Float = 0.0
    For normPos5:Int = 0 Until EMBEDDING_SIZE
        meanVal2 :+ inputData5[normPos5]
    Next
    meanVal2 :/ EMBEDDING_SIZE
    
    For normPos6:Int = 0 Until EMBEDDING_SIZE
        varianceVal2 :+ (inputData5[normPos6] - meanVal2) * (inputData5[normPos6] - meanVal2)
    Next
    varianceVal2 :/ EMBEDDING_SIZE
    
    For normPos7:Int = 0 Until EMBEDDING_SIZE
        gradX1[normPos7] = gradNormInput1[normPos7] * norm1[normPos7] / Sqr(varianceVal2 + 0.0001)
        If gradNorm1Data <> Null Then gradNorm1Data[normPos7] :+ gradNormInput1[normPos7] * (inputData5[normPos7] - meanVal2) / Sqr(varianceVal2 + 0.0001)
    Next
    
    Local meanVal3:Float = 0.0
    Local varianceVal3:Float = 0.0
    For normPos8:Int = 0 Until EMBEDDING_SIZE
        meanVal3 :+ attnOutData[normPos8]
    Next
    meanVal3 :/ EMBEDDING_SIZE
    
    For normPos9:Int = 0 Until EMBEDDING_SIZE
        varianceVal3 :+ (attnOutData[normPos9] - meanVal3) * (attnOutData[normPos9] - meanVal3)
    Next
    varianceVal3 :/ EMBEDDING_SIZE
    
    For normPos10:Int = 0 Until EMBEDDING_SIZE
        gradX2[normPos10] = gradNormInput2[normPos10] * norm2[normPos10] / Sqr(varianceVal3 + 0.0001)
        If gradNorm2Data <> Null Then gradNorm2Data[normPos10] :+ gradNormInput2[normPos10] * (attnOutData[normPos10] - meanVal3) / Sqr(varianceVal3 + 0.0001)
    Next
    
    If gradInputData3 <> Null
        For normPos11:Int = 0 Until EMBEDDING_SIZE
            gradInputData3[normPos11] :+ gradX1[normPos11] + gradX2[normPos11]
        Next
    EndIf
End Method
End Type

Type TAdvancedLLM
    Field tokenDB:String[]
    Field tokenCounts:Int[]
    Field embeddingWeights:Float[]
    Field memory:THierarchicalMemory
    Field totalInteractions:Long
    Field learningEnabled:Int
    Field enginePrompt:String
    Field conversationHistory:String[]
    Field currentContext:Float[]
    Field temperature:Float
    Field bpePairs:TBpePair[]
    Field batchBuffer:TList
    Field lastCheckpoint:Int
    Field transformerBlocks:TTransformerBlock[]
    
    Method New()
        tokenDB = New String[MAX_TOKENS]
        tokenCounts = New Int[MAX_TOKENS]
        embeddingWeights = New Float[MAX_TOKENS * EMBEDDING_SIZE]
        memory = New THierarchicalMemory
        currentContext = New Float[EMBEDDING_SIZE]
        bpePairs = New TBpePair[0]
        batchBuffer = CreateList()
        learningEnabled = IS_LEARNING
        temperature = TEMPERATURE
        
        transformerBlocks = New TTransformerBlock[NUM_LAYERS]
        For layerIdxVal:Int = 0 Until NUM_LAYERS
            transformerBlocks[layerIdxVal] = New TTransformerBlock
        Next
        
        InitializeSpecialTokens()
        InitializeWeights()
        ClearContext()
    End Method

Method Free()
    ' Limpiar arrays grandes
    embeddingWeights = Null
    tokenDB = Null
    tokenCounts = Null
    
    ' Limpiar bloques transformer
    For bloqueIdxUnico:Int = 0 Until NUM_LAYERS
        transformerBlocks[bloqueIdxUnico].attention.qWeights = Null
        transformerBlocks[bloqueIdxUnico].attention.kWeights = Null
        transformerBlocks[bloqueIdxUnico].attention.vWeights = Null
        transformerBlocks[bloqueIdxUnico].attention.oWeights = Null
        
        transformerBlocks[bloqueIdxUnico].ffn.weights1 = Null
        transformerBlocks[bloqueIdxUnico].ffn.weights2 = Null
        transformerBlocks[bloqueIdxUnico].ffn.bias1 = Null
        transformerBlocks[bloqueIdxUnico].ffn.bias2 = Null
    Next
    transformerBlocks = Null
    
    ' Limpiar memoria y contexto
    memory.shortTerm = Null
    memory.mediumTerm = Null
    memory.longTerm = Null
    memory.importanceScores = Null
    currentContext = Null
    
    ' Limpiar historial
    conversationHistory = Null
    bpePairs = Null
    
    ' Limpiar buffer de batch
    batchBuffer.Clear()
End Method

Method ImprimirEmbedding(tokenStr:String)
    Local tokenID:Int = ObtenerOCrearID(tokenStr)
    Local embFile:TStream = WriteFile("Embeddings.txt")
    If tokenID = TK_UNKNOWN
        Print "Token desconocido: " + tokenStr
        WriteLine embFile, "Token desconocido: "+tokenStr
        Return
    EndIf
    Print "Embedding para el token '" + tokenStr + "' (ID: " + tokenID + "):"
    WriteLine embFile, "Embedding para el token '" + tokenStr + "' (ID: " + tokenID + "):"
    For embiPos:Int = 0 Until EMBEDDING_SIZE
        Print "  Dimension " + embiPos + ": " + embeddingWeights[tokenID * EMBEDDING_SIZE + embiPos]
        WriteLine embFile, "  Dimension " + embiPos + ": " + embeddingWeights[tokenID * EMBEDDING_SIZE + embiPos]
    Next
    CloseFile(embFile)
End Method

Method RegistrarLog(mensajeLog:String, tipoLog:String = "INFO")
    If Not LOG_ACTIVADO Then Return
    
    Local marcaTemporal:String = CurrentTime().ToString() + " " + tipoLog + " - "
    Local lineaCompleta:String = marcaTemporal + mensajeLog
    
    ' Escribir en consola
    Print lineaCompleta
    
    ' Escribir en archivo con append manual
    If LOG_TO_FILE
        Local archivoLog:TStream
        Local existeArchivo:Int = FileType(LOG_FILE) = FILETYPE_FILE
        
        ' Abrir archivo en modo append o crear nuevo
        If existeArchivo
            archivoLog = OpenFile(LOG_FILE)
            If archivoLog
                SeekStream(archivoLog, StreamSize(archivoLog)) ' Mover al final
            EndIf
        Else
            archivoLog = WriteFile(LOG_FILE)
        EndIf
        
        If archivoLog
            archivoLog.WriteLine(lineaCompleta)
            archivoLog.Close()
        Else
            Print "Error crítico: No se pudo abrir/escribir en " + LOG_FILE
        EndIf
    EndIf
End Method

    Method InitializeSpecialTokens()
        tokenDB[TK_UNKNOWN] = "[UNK]"
        tokenDB[TK_START] = "[START]"
        tokenDB[TK_END] = "[END]"
        tokenDB[TK_RES] = "[RES]"
        tokenDB[TK_PROMPT] = "[PROMPT]"
        tokenDB[TK_NEWLINE] = "~n"
        tokenDB[TK_PAD] = "[PAD]"
        
        For specialTokenPos:Int = 0 To 6
            tokenCounts[specialTokenPos] = 1
        Next
    End Method
    
    Method InitializeWeights()
        Local scaleVal4:Float = Sqr(6.0 / (MAX_TOKENS + EMBEDDING_SIZE))
        For tokenPosVal:Int = 0 Until MAX_TOKENS
            For embPosVal:Int = 0 Until EMBEDDING_SIZE
                embeddingWeights[tokenPosVal * EMBEDDING_SIZE + embPosVal] = Rnd(-scaleVal4, scaleVal4)
            Next
        Next
    End Method
    
    Method SetTemperature(tempVal:Float)
        temperature = Max(0.1, Min(2.0, tempVal))
    End Method
    
    Method SetEnginePrompt(promptVal:String)
        enginePrompt = promptVal
        If promptVal <> ""
            If conversationHistory.Length = 0
                conversationHistory :+ ["System: " + promptVal]
            Else
                conversationHistory[0] = "System: " + promptVal
            EndIf
        Else
            If conversationHistory.Length > 0 And conversationHistory[0].StartsWith("System:")
                conversationHistory = conversationHistory[1..]
            EndIf
        EndIf
    End Method
    
    Method AddToHistory(messageVal:String, isUserFlag:Int)
        If isUserFlag
            conversationHistory :+ ["User: " + messageVal]
        Else
            conversationHistory :+ ["AI: " + messageVal]
        EndIf
        
        If conversationHistory.Length > 24
            Local newHistoryArr:String[]
            If enginePrompt <> "" And conversationHistory.Length > 0 And conversationHistory[0].StartsWith("System:")
                newHistoryArr :+ [conversationHistory[0]]
            End If
            
            Local importanceArr:Float[] = New Float[conversationHistory.Length]
            Local totalScoreVal:Float = 0
            
            For histPosVal:Int = 0 Until conversationHistory.Length
                If enginePrompt <> "" And histPosVal = 0 And conversationHistory[0].StartsWith("System:") Then Continue
                
                importanceArr[histPosVal] = CalculateImportance(conversationHistory[histPosVal])
                totalScoreVal :+ importanceArr[histPosVal]
            Next
            
            For histPosVal2:Int = 0 Until conversationHistory.Length
                If enginePrompt <> "" And histPosVal2 = 0 And conversationHistory[0].StartsWith("System:") Then Continue
                
                If importanceArr[histPosVal2] > totalScoreVal/conversationHistory.Length Or histPosVal2 >= conversationHistory.Length-4
                    newHistoryArr :+ [conversationHistory[histPosVal2]]
                EndIf
            Next
            
            conversationHistory = newHistoryArr
        EndIf
    End Method
    
    Method CalculateImportance:Float(messageStr:String)
        Local scoreVal:Float = 0.2
        scoreVal :+ 0.0005 * messageStr.Length
        If messageStr.Find("?") >= 0 Then scoreVal :+ 0.3
        If messageStr.Find("!") >= 0 Then scoreVal :+ 0.2
        If messageStr.Find("http://") >=0 Or messageStr.Find("https://") >=0 Then scoreVal :+ 0.5
        If messageStr.StartsWith("System:") Then scoreVal :+ 1.0
        Return Min(2.0, scoreVal)
    End Method
    
    Method ClearContext()
        memory.Clear()
        If enginePrompt <> ""
            conversationHistory = ["System: " + enginePrompt]
        Else
            conversationHistory = New String[0]
        EndIf
    End Method
    
    Method EnableLearning(enableFlag:Int)
        learningEnabled = enableFlag
    End Method
Rem
Method Tokenizar:String[](tkz_textoEntrada:String)
    ' Caracteres especiales definidos con Chr()
    Local tkz_especiales:String = Chr(33) + Chr(34) + Chr(35) + Chr(36) + Chr(37) + Chr(38) + Chr(39) + ..
                                 Chr(40) + Chr(41) + Chr(42) + Chr(43) + Chr(44) + Chr(45) + Chr(46) + ..
                                 Chr(47) + Chr(58) + Chr(59) + Chr(60) + Chr(61) + Chr(62) + Chr(63) + ..
                                 Chr(64) + Chr(91) + Chr(92) + Chr(93) + Chr(94) + Chr(95) + Chr(96) + ..
                                 Chr(123) + Chr(124) + Chr(125) + Chr(126)
    
    Local tkz_resultado:TList = CreateList()
    Local tkz_bufferActual:String = ""
    Local tkz_estadoPalabra:Int = False
    Local tkz_estadoNumero:Int = False
    Local tkz_posicion:Int = 0
    
    While tkz_posicion < tkz_textoEntrada.Length
        Local tkz_codigo:Int = tkz_textoEntrada[tkz_posicion]
        Local tkz_caracter:String = Chr(tkz_codigo)
        
        ' Definición de categorías
        Local tkz_esLetra:Int = (tkz_codigo >= 97 And tkz_codigo <= 122) Or (tkz_codigo >= 65 And tkz_codigo <= 90)
        Local tkz_esNumero:Int = tkz_codigo >= 48 And tkz_codigo <= 57
        Local tkz_esEspacio:Int = tkz_codigo = 32 Or tkz_codigo = 9
        Local tkz_esNuevaLinea:Int = tkz_codigo = 10
        Local tkz_esEspecial:Int = tkz_especiales.Find(tkz_caracter) >= 0
        Local tkz_esApostrofe:Int = tkz_caracter = Chr(39)
        Local tkz_esGuion:Int = tkz_caracter = Chr(45)
        
        ' Manejo de estados
        If tkz_estadoPalabra And (tkz_esLetra Or tkz_esApostrofe Or tkz_esGuion)
            tkz_bufferActual :+ tkz_caracter.ToLower()
            tkz_posicion :+ 1
            Continue
        ElseIf tkz_estadoNumero And (tkz_esNumero Or tkz_caracter = Chr(46) Or tkz_caracter = Chr(44))
            tkz_bufferActual :+ tkz_caracter
            tkz_posicion :+ 1
            Continue
        EndIf
        
        ' Flush buffer si hay contenido
        If tkz_bufferActual.Length > 0
            ListAddLast(tkz_resultado, tkz_bufferActual)
            tkz_bufferActual = ""
            tkz_estadoPalabra = False
            tkz_estadoNumero = False
        EndIf
        
        ' Manejo de caracteres individuales
        If tkz_esLetra
            tkz_estadoPalabra = True
            tkz_bufferActual :+ tkz_caracter.ToLower()
        ElseIf tkz_esNumero
            tkz_estadoNumero = True
            tkz_bufferActual :+ tkz_caracter
        ElseIf tkz_esEspacio
            ListAddLast(tkz_resultado, " ")
        ElseIf tkz_esNuevaLinea
            ListAddLast(tkz_resultado, "~n")
        ElseIf tkz_esEspecial
            ListAddLast(tkz_resultado, tkz_caracter)
        ElseIf tkz_codigo > 127 ' Caracteres Unicode
            ListAddLast(tkz_resultado, tkz_caracter)
        EndIf
        
        tkz_posicion :+ 1
    Wend
    
    ' Asegurar que el buffer final se vacíe
    If tkz_bufferActual.Length > 0
        ListAddLast(tkz_resultado, tkz_bufferActual)
    EndIf
    
    ' Convertir lista a array y limpiar tokens vacíos
    Local tkz_arrayResultado:String[] = New String[tkz_resultado.Count()]
    Local tkz_indice:Int = 0
    
    For tkz_token:String = EachIn tkz_resultado
        If tkz_token.Trim() <> ""
            tkz_arrayResultado[tkz_indice] = tkz_token
            tkz_indice :+ 1
        EndIf
    Next
    
    Return tkz_arrayResultado[..tkz_indice]
End Method
EndRem
Method Tokenizar:String[](tkn_textoEntrada:String)
    ' --- Variables locales con prefijo tkn_ ---
    Local tkn_caracteresEspeciales:String = Chr(33)+Chr(34)+Chr(35)+Chr(36)+Chr(37)+Chr(38)+Chr(39)+Chr(40)+Chr(41)+..
        Chr(42)+Chr(43)+Chr(44)+Chr(45)+Chr(46)+Chr(47)+Chr(58)+Chr(59)+Chr(60)+Chr(61)+Chr(62)+Chr(63)+Chr(64)+..
        Chr(91)+Chr(92)+Chr(93)+Chr(94)+Chr(95)+Chr(96)+Chr(123)+Chr(124)+Chr(125)+Chr(126)
    
    ' --- Paso 1: Tokenización inicial ---
    Local tkn_tokensIniciales:TList = CreateList()
    Local tkn_bufferActual:String = ""
    Local tkn_enPalabra:Int = False
    Local tkn_enNumero:Int = False
    
    For tkn_i:Int = 0 Until tkn_textoEntrada.Length
        Local tkn_c:Int = tkn_textoEntrada[tkn_i]
        Local tkn_char:String = Chr(tkn_c)
        
        ' Categorización
        Local tkn_esLetra:Int = (tkn_c >= 97 And tkn_c <= 122) Or (tkn_c >= 65 And tkn_c <= 90)
        Local tkn_esNumero:Int = tkn_c >= 48 And tkn_c <= 57
        Local tkn_esEspacio:Int = tkn_c = 32 Or tkn_c = 9 Or tkn_c = 10 Or tkn_c = 13
        Local tkn_esEspecial:Int = tkn_caracteresEspeciales.Find(tkn_char) >= 0
        Local tkn_esApostrofe:Int = tkn_char = Chr(39)
        Local tkn_esGuion:Int = tkn_char = Chr(45)
        
        ' Manejo de estados
        If tkn_enPalabra And (tkn_esLetra Or tkn_esApostrofe Or tkn_esGuion)
            tkn_bufferActual :+ tkn_char.ToLower()
            Continue
        ElseIf tkn_enNumero And (tkn_esNumero Or tkn_char = Chr(46) Or tkn_char = Chr(44))
            tkn_bufferActual :+ tkn_char
            Continue
        EndIf
        
        ' Flush buffer si hay contenido
        If tkn_bufferActual.Length > 0
            ListAddLast(tkn_tokensIniciales, tkn_bufferActual)
            tkn_bufferActual = ""
            tkn_enPalabra = False
            tkn_enNumero = False
        EndIf
        
        ' Manejo de caracteres individuales
        If tkn_esLetra
            tkn_enPalabra = True
            tkn_bufferActual :+ tkn_char.ToLower()
        ElseIf tkn_esNumero
            tkn_enNumero = True
            tkn_bufferActual :+ tkn_char
        ElseIf tkn_esEspacio
            ListAddLast(tkn_tokensIniciales, " ")
        ElseIf tkn_esEspecial
            ListAddLast(tkn_tokensIniciales, tkn_char)
        ElseIf tkn_c > 127 ' Caracteres Unicode
            ListAddLast(tkn_tokensIniciales, tkn_char)
        EndIf
    Next
    
    ' Asegurar que el buffer final se vacíe
    If tkn_bufferActual.Length > 0
        ListAddLast(tkn_tokensIniciales, tkn_bufferActual)
    EndIf
    
    ' --- Paso 2: Convertir a array y aplicar BPE ---
    Local tkn_tokensArray:String[] = New String[tkn_tokensIniciales.Count()]
    Local tkn_idx:Int = 0
    
    For tkn_token:String = EachIn tkn_tokensIniciales
        If tkn_token.Trim() <> ""
            tkn_tokensArray[tkn_idx] = tkn_token
            tkn_idx :+ 1
        EndIf
    Next
    
    tkn_tokensArray = tkn_tokensArray[..tkn_idx]
    
    ' --- Paso 3: Aplicar BPE dinámico ---
    Local tkn_resultadoFinal:TList = CreateList()
    
    For tkn_tokenActual:String = EachIn tkn_tokensArray
        If tkn_tokenActual.Length = 1 Or tkn_tokenActual = " " Or tkn_caracteresEspeciales.Find(tkn_tokenActual) >= 0
            ListAddLast(tkn_resultadoFinal, tkn_tokenActual)
            Continue
        EndIf
        
        ' Descomponer en caracteres (reemplazo de ToCString)
        Local tkn_caracteres:String[] = New String[tkn_tokenActual.Length]
        For tkn_b:Int = 0 Until tkn_tokenActual.Length
            tkn_caracteres[tkn_b] = Chr(tkn_tokenActual[tkn_b])
        Next
        
        ' Aplicar merges BPE
        Local tkn_cambios:Int = True
        While tkn_cambios
            tkn_cambios = False
            Local tkn_mejorPar:String = ""
            Local tkn_mejorFrecuencia:Int = 0
            Local tkn_mejorPos:Int = -1
            
            ' Buscar el par más frecuente
            For tkn_j:Int = 0 Until tkn_caracteres.Length-1
                Local tkn_par:String = tkn_caracteres[tkn_j] + tkn_caracteres[tkn_j+1]
                Local tkn_id:Int = ObtenerOCrearID(tkn_par)
                If tkn_id <> TK_UNKNOWN And tokenCounts[tkn_id] > tkn_mejorFrecuencia
                    tkn_mejorFrecuencia = tokenCounts[tkn_id]
                    tkn_mejorPar = tkn_par
                    tkn_mejorPos = tkn_j
                EndIf
            Next
            
            ' Aplicar merge si encontramos un par válido
            If tkn_mejorPos >= 0
                ' Reemplazar el par por el token combinado
                Local tkn_nuevosCaracteres:String[] = New String[tkn_caracteres.Length-1]
                For tkn_k:Int = 0 Until tkn_mejorPos
                    tkn_nuevosCaracteres[tkn_k] = tkn_caracteres[tkn_k]
                Next
                tkn_nuevosCaracteres[tkn_mejorPos] = tkn_mejorPar
                For tkn_k:Int = tkn_mejorPos+1 Until tkn_nuevosCaracteres.Length
                    tkn_nuevosCaracteres[tkn_k] = tkn_caracteres[tkn_k+1]
                Next
                
                tkn_caracteres = tkn_nuevosCaracteres
                tkn_cambios = True
                
                ' Actualizar contador para este par
                Local tkn_parId:Int = ObtenerOCrearID(tkn_mejorPar)
                tokenCounts[tkn_parId] :+ 1
            EndIf
        Wend
        
        ' Agregar subtokens al resultado final
        For tkn_subtoken:String = EachIn tkn_caracteres
            ListAddLast(tkn_resultadoFinal, tkn_subtoken)
        Next
    Next
    
    ' Convertir lista final a array
    Local tkn_arrayFinal:String[] = New String[tkn_resultadoFinal.Count()]
    Local tkn_finalIdx:Int = 0
    
    For tkn_tokenFinal:String = EachIn tkn_resultadoFinal
        tkn_arrayFinal[tkn_finalIdx] = tkn_tokenFinal
        tkn_finalIdx :+ 1
    Next
    
    Return tkn_arrayFinal
End Method

    Method ObtenerOCrearID:Int(tokenStr:String)
        Select tokenStr
            Case "[UNK]" Return TK_UNKNOWN
            Case "[START]" Return TK_START
            Case "[END]" Return TK_END
            Case "[RES]" Return TK_RES
            Case "[PROMPT]" Return TK_PROMPT
            Case "~n" Return TK_NEWLINE
            Case "[PAD]" Return TK_PAD
        End Select
        
        For tokenIDVal:Int = 7 Until MAX_TOKENS
            If tokenDB[tokenIDVal] = tokenStr
                Return tokenIDVal
            EndIf
        Next
        
        For newIDVal:Int = 7 Until MAX_TOKENS
            If tokenDB[newIDVal] = Null Or tokenDB[newIDVal] = ""
                tokenDB[newIDVal] = tokenStr
                tokenCounts[newIDVal] = 1
                
                Local similarFlag:Int = False
                For embPosVal2:Int = 0 Until EMBEDDING_SIZE
                    Local sumVal2:Float = 0
                    Local countVal:Int = 0
                    
                    For subPosVal:Int = 0 Until tokenStr.Length-1
                        Local subTokenStr:String = tokenStr[subPosVal..subPosVal+1]
                        Local subIDVal:Int = ObtenerOCrearID(subTokenStr)
                        If subIDVal <> TK_UNKNOWN
                            sumVal2 :+ embeddingWeights[subIDVal * EMBEDDING_SIZE + embPosVal2]
                            countVal :+ 1
                        EndIf
                    Next
                    
                    If countVal > 0
                        embeddingWeights[newIDVal * EMBEDDING_SIZE + embPosVal2] = sumVal2 / countVal
                        similarFlag = True
                    Else
                        Local scaleVal5:Float = Sqr(2.0 / (newIDVal+1 + EMBEDDING_SIZE))
                        embeddingWeights[newIDVal * EMBEDDING_SIZE + embPosVal2] = Rnd(-scaleVal5, scaleVal5)
                    EndIf
                Next
                
                Return newIDVal
            EndIf
        Next
        
        Return TK_UNKNOWN
    End Method

Method ProcesarEntrada:Float[](arregloTokens:Int[])
    Local contextoAcumulado:Float[] = New Float[EMBEDDING_SIZE]
    Local memoriaTemporal:Float[] = New Float[EMBEDDING_SIZE]
    
    ' Procesar por bloques para mejor eficiencia
    For inicioBloque:Int = 0 Until arregloTokens.Length Step BATCH_SIZE
        Local finBloque:Int = Min(inicioBloque + BATCH_SIZE, arregloTokens.Length)
        
        For posicionToken:Int = inicioBloque Until finBloque
            Local idTokenActual:Int = arregloTokens[posicionToken]
            Local representacionToken:Float[] = New Float[EMBEDDING_SIZE]
            
            ' Obtener embedding base
            For dimensionEmbedding:Int = 0 Until EMBEDDING_SIZE
                representacionToken[dimensionEmbedding] = embeddingWeights[idTokenActual * EMBEDDING_SIZE + dimensionEmbedding]
            Next
            
            ' Aplicar capas de transformación
            For indiceCapa:Int = 0 Until NUM_LAYERS
                representacionToken = transformerBlocks[indiceCapa].Process(representacionToken)
            Next
            
            ' Acumular en el contexto
            For dimensionAcumulacion:Int = 0 Until EMBEDDING_SIZE
                contextoAcumulado[dimensionAcumulacion] :+ representacionToken[dimensionAcumulacion] * 0.6
                memoriaTemporal[dimensionAcumulacion] :+ representacionToken[dimensionAcumulacion] * 0.1
            Next
            
            ' Actualizar contador de tokens
            tokenCounts[idTokenActual] :+ 1
        Next
        
        ' Normalización intermedia
        Local magnitudContexto:Float = 0.0
        For dimensionNormalizacion:Int = 0 Until EMBEDDING_SIZE
            magnitudContexto :+ contextoAcumulado[dimensionNormalizacion] * contextoAcumulado[dimensionNormalizacion]
        Next
        magnitudContexto = Sqr(magnitudContexto)
        
        If magnitudContexto > 0.0
            For dimensionAjuste:Int = 0 Until EMBEDDING_SIZE
                contextoAcumulado[dimensionAjuste] :/ magnitudContexto
            Next
        EndIf
    Next
    
    ' Actualizar memoria a largo plazo
    memory.UpdateMemory(memoriaTemporal)
    
    ' Combinar con memoria existente
    Local contextoFinal:Float[] = memory.GetContext()
    For dimensionCombinacion:Int = 0 Until EMBEDDING_SIZE
        contextoFinal[dimensionCombinacion] = 0.7 * contextoFinal[dimensionCombinacion] + 0.3 * contextoAcumulado[dimensionCombinacion]
    Next
    
    ' Normalización final
    Local magnitudFinal:Float = 0.0
    For dimensionFinal:Int = 0 Until EMBEDDING_SIZE
        magnitudFinal :+ contextoFinal[dimensionFinal] * contextoFinal[dimensionFinal]
    Next
    magnitudFinal = Sqr(magnitudFinal)
    
    If magnitudFinal > 0.0
        For dimensionNormalizada:Int = 0 Until EMBEDDING_SIZE
            contextoFinal[dimensionNormalizada] :/ magnitudFinal
            currentContext[dimensionNormalizada] = 0.9 * currentContext[dimensionNormalizada] + 0.1 * contextoFinal[dimensionNormalizada]
        Next
    EndIf
    
    Return currentContext
End Method

    Method Backpropagate(inputIDsArr:Int[], outputIDsArr:Int[], contextArr2:Float[])
        Local gradEmbedArr:Float[] = New Float[MAX_TOKENS * EMBEDDING_SIZE]
        
        For outPosVal:Int = 0 Until outputIDsArr.Length
            Local tokenIDVal3:Int = outputIDsArr[outPosVal]
            For embPosVal9:Int = 0 Until EMBEDDING_SIZE
                Local errorVal:Float = contextArr2[embPosVal9] - embeddingWeights[tokenIDVal3 * EMBEDDING_SIZE + embPosVal9]
                gradEmbedArr[tokenIDVal3 * EMBEDDING_SIZE + embPosVal9] :+ LEARNING_RATE * errorVal / outputIDsArr.Length
            Next
        Next
        
        Local gradInputArr:Float[] = New Float[EMBEDDING_SIZE]
        For inPosVal:Int = 0 Until inputIDsArr.Length
            Local tokenIDVal4:Int = inputIDsArr[inPosVal]
            Local tokenEmbedArr2:Float[] = New Float[EMBEDDING_SIZE]
            
            For embPosVal10:Int = 0 Until EMBEDDING_SIZE
                tokenEmbedArr2[embPosVal10] = embeddingWeights[tokenIDVal4 * EMBEDDING_SIZE + embPosVal10]
            Next
            
            ' Reemplazo de arrays multidimensionales con arrays planos
            Local layerGradientsFlat:Float[] = New Float[NUM_LAYERS * EMBEDDING_SIZE]
            Local layerInputsFlat:Float[] = New Float[(NUM_LAYERS+1) * EMBEDDING_SIZE]
            
            ' Copiar input inicial
            For embCopyPos:Int = 0 Until EMBEDDING_SIZE
                layerInputsFlat[0 * EMBEDDING_SIZE + embCopyPos] = tokenEmbedArr2[embCopyPos]
            Next

            For layerIdxVal3:Int = 0 Until NUM_LAYERS
                Local layerOutput:Float[] = transformerBlocks[layerIdxVal3].Process(layerInputsFlat[layerIdxVal3 * EMBEDDING_SIZE..(layerIdxVal3+1)*EMBEDDING_SIZE])
                
                ' Copiar output al siguiente input
                For embCopyPos2:Int = 0 Until EMBEDDING_SIZE
                    layerInputsFlat[(layerIdxVal3+1) * EMBEDDING_SIZE + embCopyPos2] = layerOutput[embCopyPos2]
                Next
            Next
            
            Local gradOutputArr:Float[] = New Float[EMBEDDING_SIZE]
            For embPosVal11:Int = 0 Until EMBEDDING_SIZE
                gradOutputArr[embPosVal11] = contextArr2[embPosVal11]
            Next

            For layerIdxVal4:Int = NUM_LAYERS-1 To 0 Step -1
                Local gradLayerInputArr:Float[] = New Float[EMBEDDING_SIZE]
                Local gradNorm1Arr:Float[] = New Float[EMBEDDING_SIZE]
                Local gradNorm2Arr:Float[] = New Float[EMBEDDING_SIZE]
                
                transformerBlocks[layerIdxVal4].Backpropagate(..
                    layerInputsFlat[layerIdxVal4 * EMBEDDING_SIZE..(layerIdxVal4+1)*EMBEDDING_SIZE], ..
                    gradOutputArr, ..
                    gradLayerInputArr, ..
                    gradNorm1Arr, ..
                    gradNorm2Arr)
                
                For embPosVal12:Int = 0 Until EMBEDDING_SIZE
                    gradOutputArr[embPosVal12] = gradLayerInputArr[embPosVal12]
                Next
            Next
            
            For embPosVal13:Int = 0 Until EMBEDDING_SIZE
                gradEmbedArr[tokenIDVal4 * EMBEDDING_SIZE + embPosVal13] :+ gradOutputArr[embPosVal13]
            Next
        Next
        
        For tokenIdxVal:Int = 0 Until MAX_TOKENS
            For embPosVal14:Int = 0 Until EMBEDDING_SIZE
                embeddingWeights[tokenIdxVal * EMBEDDING_SIZE + embPosVal14] :+ gradEmbedArr[tokenIdxVal * EMBEDDING_SIZE + embPosVal14]
                embeddingWeights[tokenIdxVal * EMBEDDING_SIZE + embPosVal14] :* 0.9999
            Next
        Next
    End Method
    
    Method EntrenarLLM:Int(preguntaStr:String, respuestaStr:String, verboseFlag:Int = False)
        Try
            If preguntaStr.Trim().Length = 0 Or respuestaStr.Trim().Length = 0
                If verboseFlag Then Print "Error: Pregunta o respuesta vacía"
                Return False
            EndIf
            
            Local inputTokensArr:String[] = Tokenizar(preguntaStr)
            Local outputTokensArr:String[] = Tokenizar(respuestaStr)
            
            If inputTokensArr.Length = 0 Or outputTokensArr.Length = 0
                If verboseFlag Then Print "Error: No se generaron tokens válidos"
                Return False
            EndIf
            
            If verboseFlag
                Print "Entrenando con:"
                Print "  Pregunta ("+inputTokensArr.Length+" tokens): " + preguntaStr
                Print "  Respuesta ("+outputTokensArr.Length+" tokens): " + respuestaStr
                Print "  Tokens únicos en DB: " + ContarTokensActivos()
            EndIf
            
            Local inputIDsArr2:Int[] = New Int[inputTokensArr.Length]
            Local outputIDsArr2:Int[] = New Int[outputTokensArr.Length]
            
            For tokenPosVal3:Int = 0 Until inputTokensArr.Length
                inputIDsArr2[tokenPosVal3] = ObtenerOCrearID(inputTokensArr[tokenPosVal3])
            Next
            
            For tokenPosVal4:Int = 0 Until outputTokensArr.Length
                outputIDsArr2[tokenPosVal4] = ObtenerOCrearID(outputTokensArr[tokenPosVal4])
            Next
            
            Local contextArr3:Float[] = ProcesarEntrada(inputIDsArr2)
            If learningEnabled
                Backpropagate(inputIDsArr2, outputIDsArr2, contextArr3)
                totalInteractions :+ 1
                
                If totalInteractions - lastCheckpoint >= CHECKPOINT_EVERY
                    Local nombreArchivoStr:String = "checkpoint_"+totalInteractions+".llm"
                    GuardarModelo(nombreArchivoStr, False)
                    lastCheckpoint = totalInteractions
                    If verboseFlag Then Print "Checkpoint guardado: " + nombreArchivoStr
                EndIf
            EndIf
            
            Return True
            
        Catch err:Object
            Print "Error crítico en EntrenarLLM: " + err.ToString()
            Return False
        End Try
    End Method

Method GenerarRespuesta:String(inputUsuario:String)
    ' --- Variables Locales únicas ---
    Local listaTokensInputUsuarioActual:String[] = Tokenizar(inputUsuario)
    Local arregloIdentificadoresTokensUsuario:Int[] = New Int[listaTokensInputUsuarioActual.Length]
    Local vectorContextoActualizadoModelo:Float[] = New Float[EMBEDDING_SIZE]
    Local cadenaRespuestaGenerada:String = ""
    Local identificadorTokenActualGeneracion:Int = TK_START
    Local contadorTotalTokensGenerados:Int = 0
    Local bufferCircularTokensRecientes:Int[] = [0,0,0,0]
    Local factorAleatoriedadTemperatura:Float = temperature
    Local banderaGeneracionActiva:Int = True
    Local acumuladorProbabilidadTotal:Float = 0.0
    Local puntajeSimilitudMaximaEncontrada:Float = -1.0
    Local identificadorTokenSeleccionado:Int = -1

    ' --- Procesamiento de Input ---
    For posicionTokenInputActual:Int = 0 Until listaTokensInputUsuarioActual.Length
        arregloIdentificadoresTokensUsuario[posicionTokenInputActual] = ObtenerOCrearID(listaTokensInputUsuarioActual[posicionTokenInputActual])
    Next

    vectorContextoActualizadoModelo = ProcesarEntrada(arregloIdentificadoresTokensUsuario)

    ' --- Ciclo Principal de Generación ---
    While banderaGeneracionActiva
        ' --- Configuración de Probabilidades ---
        Local arregloDistribucionProbabilidades:Float[MAX_TOKENS]
        puntajeSimilitudMaximaEncontrada = -1.0
        identificadorTokenSeleccionado = -1
        acumuladorProbabilidadTotal = 0.0

        ' --- Cálculo de Similitudes ---
        For identificadorTokenEvaluado:Int = 0 Until MAX_TOKENS
            If tokenDB[identificadorTokenEvaluado] And tokenDB[identificadorTokenEvaluado] <> ""
                ' --- Cálculo de Puntaje Base ---
                Local puntajeSimilitudActual:Float = 0.0
                For dimensionEmbeddingActual:Int = 0 Until EMBEDDING_SIZE
                    puntajeSimilitudActual = puntajeSimilitudActual + vectorContextoActualizadoModelo[dimensionEmbeddingActual] * embeddingWeights[identificadorTokenEvaluado*EMBEDDING_SIZE+dimensionEmbeddingActual]
                    puntajeSimilitudActual = puntajeSimilitudActual + embeddingWeights[identificadorTokenActualGeneracion*EMBEDDING_SIZE+dimensionEmbeddingActual] * embeddingWeights[identificadorTokenEvaluado*EMBEDDING_SIZE+dimensionEmbeddingActual] * 0.3
                Next

                ' --- Aplicar Penalizaciones ---
                Local factorPenalizacionRepeticion:Float = 1.0
                For posicionBufferCircular:Int = 0 Until bufferCircularTokensRecientes.Length
                    If bufferCircularTokensRecientes[posicionBufferCircular] = identificadorTokenEvaluado
                        factorPenalizacionRepeticion = factorPenalizacionRepeticion * 0.35
                    EndIf
                Next

                ' --- Aplicar Temperatura ---
                arregloDistribucionProbabilidades[identificadorTokenEvaluado] = Exp(puntajeSimilitudActual * factorPenalizacionRepeticion / factorAleatoriedadTemperatura)
                acumuladorProbabilidadTotal = acumuladorProbabilidadTotal + arregloDistribucionProbabilidades[identificadorTokenEvaluado]

                If arregloDistribucionProbabilidades[identificadorTokenEvaluado] > puntajeSimilitudMaximaEncontrada
                    puntajeSimilitudMaximaEncontrada = arregloDistribucionProbabilidades[identificadorTokenEvaluado]
                    identificadorTokenSeleccionado = identificadorTokenEvaluado
                EndIf
            EndIf
        Next

        ' --- Selección del Token ---
        If factorAleatoriedadTemperatura > 0.1 And acumuladorProbabilidadTotal > 0
            Local valorAleatorioSeleccion:Float = Rnd() * acumuladorProbabilidadTotal
            Local acumuladorProbabilidad:Float = 0.0
            For identificadorTokenProbable:Int = 0 Until MAX_TOKENS
                If arregloDistribucionProbabilidades[identificadorTokenProbable] > 0
                    acumuladorProbabilidad = acumuladorProbabilidad + arregloDistribucionProbabilidades[identificadorTokenProbable]
                    If acumuladorProbabilidad >= valorAleatorioSeleccion
                        identificadorTokenSeleccionado = identificadorTokenProbable
                        Exit
                    EndIf
                EndIf
            Next
        EndIf

        ' --- Manejo de Token Seleccionado ---
        If identificadorTokenSeleccionado = -1
            banderaGeneracionActiva = False
            Continue
        EndIf

        ' --- Actualización de Buffer Circular ---
        For posicionRotacionBuffer:Int = bufferCircularTokensRecientes.Length-1 To 1 Step -1
            bufferCircularTokensRecientes[posicionRotacionBuffer] = bufferCircularTokensRecientes[posicionRotacionBuffer-1]
        Next
        bufferCircularTokensRecientes[0] = identificadorTokenSeleccionado

        ' --- Construcción de Respuesta ---
        Local textoTokenActual:String = tokenDB[identificadorTokenSeleccionado]
        If cadenaRespuestaGenerada.Length > 0 And Not textoTokenActual.StartsWith(" ") And Not IsPunctuation(textoTokenActual)
            cadenaRespuestaGenerada = cadenaRespuestaGenerada + " "
        EndIf
        cadenaRespuestaGenerada = cadenaRespuestaGenerada + textoTokenActual

        ' --- Condiciones de Terminación ---
        contadorTotalTokensGenerados = contadorTotalTokensGenerados + 1
        identificadorTokenActualGeneracion = identificadorTokenSeleccionado

        If identificadorTokenSeleccionado = TK_END Or contadorTotalTokensGenerados >= MAX_RESPONSE_LENGTH
            banderaGeneracionActiva = False
        EndIf
    Wend

    ' --- Post-Procesamiento ---
    cadenaRespuestaGenerada = cadenaRespuestaGenerada.Trim()
    If cadenaRespuestaGenerada.Length > 0
        cadenaRespuestaGenerada = cadenaRespuestaGenerada[..1].ToUpper() + cadenaRespuestaGenerada[1..]
        cadenaRespuestaGenerada = cadenaRespuestaGenerada.Replace(" .", ".")
        cadenaRespuestaGenerada = cadenaRespuestaGenerada.Replace(" ,", ",")
        cadenaRespuestaGenerada = cadenaRespuestaGenerada.Replace(" ?", "?")
        cadenaRespuestaGenerada = cadenaRespuestaGenerada.Replace(" !", "!")
        cadenaRespuestaGenerada = cadenaRespuestaGenerada.Replace(" ;", ";")
    EndIf

    Return cadenaRespuestaGenerada
End Method

    Method IsPunctuation:Int(tokenStr2:String)
        Return tokenStr2 = "." Or tokenStr2 = "," Or tokenStr2 = "!" Or tokenStr2 = "?" Or tokenStr2 = ";" Or tokenStr2 = ":"
    End Method
    
    Method ContarTokensActivos:Int()
        Local countVal2:Int = 0
        For tokenIDVal5:Int = 0 Until MAX_TOKENS
            If tokenDB[tokenIDVal5] And tokenDB[tokenIDVal5] <> "" Then countVal2 :+ 1
        Next
        Return countVal2
    End Method

Method GuardarModelo(archivoStr:String, fullSaveFlag:Int = True)
    Local streamVal:TStream = WriteStream(archivoStr)
    If Not streamVal Then Return
    
    streamVal.WriteLine("BLLMv1") ' Nueva versión con formato binario
    streamVal.WriteLine(String(learningEnabled))
    streamVal.WriteLine(String(totalInteractions))
    streamVal.WriteLine(Replace(enginePrompt, "~n", "\n"))
    
    ' Guardar tokens
    streamVal.WriteLine("TOKENS")
    Local numTokensVal:Int = 0
    For tokenIDVal6:Int = 0 Until MAX_TOKENS
        If tokenDB[tokenIDVal6] And tokenDB[tokenIDVal6] <> "" Then numTokensVal :+ 1
    Next
    streamVal.WriteLine(numTokensVal)
    
    For tokenIDVal7:Int = 0 Until MAX_TOKENS
        If tokenDB[tokenIDVal7] And tokenDB[tokenIDVal7] <> ""
            streamVal.WriteLine(tokenIDVal7 + "|" + Replace(tokenDB[tokenIDVal7], "~n", "\n") + "|" + tokenCounts[tokenIDVal7])
        EndIf
    Next
    
    ' Guardar embeddings en binario
    streamVal.WriteLine("EMBEDDINGS")
    streamVal.WriteLine(EMBEDDING_SIZE) ' Escribir tamaño del embedding
    streamVal.WriteLine(ContarTokensActivos()) ' Escribir cantidad de tokens con embeddings
    
    For tokenIDVal8:Int = 0 Until MAX_TOKENS
        If tokenDB[tokenIDVal8] And tokenDB[tokenIDVal8] <> ""
            ' Escribir ID del token (4 bytes)
            streamVal.WriteInt(tokenIDVal8)
            
            ' Escribir todos los floats del embedding en binario
            For embPosVal16:Int = 0 Until EMBEDDING_SIZE
                streamVal.WriteFloat(embeddingWeights[tokenIDVal8 * EMBEDDING_SIZE + embPosVal16])
            Next
        EndIf
    Next
    
    ' Resto del modelo (igual que antes)
    streamVal.WriteLine("TRANSFORMER")
    For layerIdxVal5:Int = 0 Until NUM_LAYERS
        streamVal.WriteLine("LAYER_" + layerIdxVal5)

        For iIdxVal:Int = 0 Until EMBEDDING_SIZE
            For jIdxVal:Int = 0 Until EMBEDDING_SIZE
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].attention.qWeights[iIdxVal * EMBEDDING_SIZE + jIdxVal])
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].attention.kWeights[iIdxVal * EMBEDDING_SIZE + jIdxVal])
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].attention.vWeights[iIdxVal * EMBEDDING_SIZE + jIdxVal])
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].attention.oWeights[iIdxVal * EMBEDDING_SIZE + jIdxVal])
            Next
        Next
        
        ' Guardar FFN (igual que antes)
        For iIdxVal2:Int = 0 Until EMBEDDING_SIZE
            For jIdxVal2:Int = 0 Until FFN_HIDDEN_SIZE
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].ffn.weights1[iIdxVal2 * FFN_HIDDEN_SIZE + jIdxVal2])
            Next
        Next
        
        For iIdxVal3:Int = 0 Until FFN_HIDDEN_SIZE
            For jIdxVal3:Int = 0 Until EMBEDDING_SIZE
                streamVal.WriteFloat(transformerBlocks[layerIdxVal5].ffn.weights2[iIdxVal3 * EMBEDDING_SIZE + jIdxVal3])
            Next
        Next
        
        For iIdxVal4:Int = 0 Until FFN_HIDDEN_SIZE
            streamVal.WriteFloat(transformerBlocks[layerIdxVal5].ffn.bias1[iIdxVal4])
        Next
        
        For iIdxVal5:Int = 0 Until EMBEDDING_SIZE
            streamVal.WriteFloat(transformerBlocks[layerIdxVal5].ffn.bias2[iIdxVal5])
        Next
        
        ' Guardar normalizaciones
        For iIdxVal6:Int = 0 Until EMBEDDING_SIZE
            streamVal.WriteFloat(transformerBlocks[layerIdxVal5].norm1[iIdxVal6])
            streamVal.WriteFloat(transformerBlocks[layerIdxVal5].norm2[iIdxVal6])
        Next
    Next
    
    If fullSaveFlag
        streamVal.WriteLine("MEMORY")
        Local memArr:Float[] = memory.GetContext()
        For embPosVal17:Int = 0 Until EMBEDDING_SIZE
            streamVal.WriteFloat(memArr[embPosVal17])
        Next
        
        streamVal.WriteLine("HISTORY")
        streamVal.WriteLine(conversationHistory.Length)
        For msgStr:String = EachIn conversationHistory
            streamVal.WriteLine(msgStr)
        Next
    EndIf
    
    streamVal.Close()
End Method

Method CargarModelo:Int(archivoStr:String, loadFullFlag:Int = False)
    Local streamVal2:TStream = ReadStream(archivoStr)
    If streamVal2=Null Or FileSize(archivoStr)=0 Then Notify("Error al cargar modelo.", 1) End ' Return False
    
    Local versionStr:String = streamVal2.ReadLine()
    If versionStr <> "BLLMv1"
        streamVal2.Close()
        Print "Error: Versión de archivo no compatible. Esperaba BLLMv1, obtuvo "+versionStr+"."
        Return False
    EndIf
    
    learningEnabled = Int(streamVal2.ReadLine())
    totalInteractions = Long(streamVal2.ReadLine())
    enginePrompt = Replace(streamVal2.ReadLine(), "\n", "~n")
    
    While Not streamVal2.EOF()
        Local sectionStr:String = streamVal2.ReadLine()
        
        Select sectionStr
            Case "TOKENS"
                Local numTokensVal2:Int = Int(streamVal2.ReadLine())
                For nVal:Int = 1 To numTokensVal2
                    Local lineaStr2:String = streamVal2.ReadLine()
                    If lineaStr2 = "" Then Continue
                    
                    Local partesArr:String[] = lineaStr2.Split("|")
                    If partesArr.Length < 3 Then Continue
                    
                    Local idVal:Int = Int(partesArr[0])
                    Local token_strVal:String = Replace(partesArr[1], "\n", "~n")
                    Local count_strVal:String = partesArr[2]
                    
                    If idVal >= 0 And idVal < MAX_TOKENS
                        tokenDB[idVal] = token_strVal
                        tokenCounts[idVal] = Int(count_strVal)
                    EndIf
                Next
                
            Case "EMBEDDINGS"
                ' Leer tamaño del embedding (debería coincidir con EMBEDDING_SIZE)
                Local embSize:Int = Int(streamVal2.ReadLine())
                If embSize <> EMBEDDING_SIZE
                    Print "Advertencia: Tamaño de embedding no coincide (" + embSize + " vs " + EMBEDDING_SIZE + ")"
                EndIf
                
                ' Leer cantidad de tokens con embeddings
                Local numEmbeddings:Int = Int(streamVal2.ReadLine())
                
                For n:Int = 1 To numEmbeddings
                    ' Leer ID del token
                    Local tokenID:Int = streamVal2.ReadInt()
                    
                    ' Leer todos los floats del embedding
                    For embPos:Int = 0 Until EMBEDDING_SIZE
                        embeddingWeights[tokenID * EMBEDDING_SIZE + embPos] = streamVal2.ReadFloat()
                    Next
                Next
                
            Case "TRANSFORMER"
                For layerIdxVal6:Int = 0 Until NUM_LAYERS
                    Local layerHeaderStr:String = streamVal2.ReadLine()
                    If layerHeaderStr <> "LAYER_" + layerIdxVal6 Then Exit

                    For iIdxVal7:Int = 0 Until EMBEDDING_SIZE
                        For jIdxVal4:Int = 0 Until EMBEDDING_SIZE
                            transformerBlocks[layerIdxVal6].attention.qWeights[iIdxVal7 * EMBEDDING_SIZE + jIdxVal4] = streamVal2.ReadFloat()
                            transformerBlocks[layerIdxVal6].attention.kWeights[iIdxVal7 * EMBEDDING_SIZE + jIdxVal4] = streamVal2.ReadFloat()
                            transformerBlocks[layerIdxVal6].attention.vWeights[iIdxVal7 * EMBEDDING_SIZE + jIdxVal4] = streamVal2.ReadFloat()
                            transformerBlocks[layerIdxVal6].attention.oWeights[iIdxVal7 * EMBEDDING_SIZE + jIdxVal4] = streamVal2.ReadFloat()
                        Next
                    Next
                    
                    ' Cargar FFN
                    For iIdxVal8:Int = 0 Until EMBEDDING_SIZE
                        For jIdxVal5:Int = 0 Until FFN_HIDDEN_SIZE
                            transformerBlocks[layerIdxVal6].ffn.weights1[iIdxVal8 * FFN_HIDDEN_SIZE + jIdxVal5] = streamVal2.ReadFloat()
                        Next
                    Next
                    
                    For iIdxVal9:Int = 0 Until FFN_HIDDEN_SIZE
                        For jIdxVal6:Int = 0 Until EMBEDDING_SIZE
                            transformerBlocks[layerIdxVal6].ffn.weights2[iIdxVal9 * EMBEDDING_SIZE + jIdxVal6] = streamVal2.ReadFloat()
                        Next
                    Next
                    
                    For iIdxVal10:Int = 0 Until FFN_HIDDEN_SIZE
                        transformerBlocks[layerIdxVal6].ffn.bias1[iIdxVal10] = streamVal2.ReadFloat()
                    Next
                    
                    For iIdxVal11:Int = 0 Until EMBEDDING_SIZE
                        transformerBlocks[layerIdxVal6].ffn.bias2[iIdxVal11] = streamVal2.ReadFloat()
                    Next
                    
                    ' Cargar normalizaciones
                    For iIdxVal12:Int = 0 Until EMBEDDING_SIZE
                        transformerBlocks[layerIdxVal6].norm1[iIdxVal12] = streamVal2.ReadFloat()
                        transformerBlocks[layerIdxVal6].norm2[iIdxVal12] = streamVal2.ReadFloat()
                    Next
                Next
                
            Case "MEMORY"
                If loadFullFlag
                    For embPosVal19:Int = 0 Until EMBEDDING_SIZE
                        memory.GetContext()[embPosVal19] = streamVal2.ReadFloat()
                    Next
                EndIf
                
            Case "HISTORY"
                If loadFullFlag
                    Local historySizeVal:Int = Int(streamVal2.ReadLine())
                    conversationHistory = New String[historySizeVal]
                    For histIdxVal4:Int = 0 Until historySizeVal
                        conversationHistory[histIdxVal4] = streamVal2.ReadLine()
                    Next
                EndIf
        End Select
    Wend
    
    streamVal2.Close()
    
    If conversationHistory.Length > 0 And loadFullFlag
        Local fullContextStr:String = ""
        For msgStr2:String = EachIn conversationHistory
            fullContextStr :+ msgStr2 + "~n"
        Next
        
        Local inputTokensArr3:String[] = Tokenizar(fullContextStr)
        Local inputIDsArr3:Int[] = New Int[inputTokensArr3.Length]
        
        For tokenPosVal6:Int = 0 Until inputTokensArr3.Length
            inputIDsArr3[tokenPosVal6] = ObtenerOCrearID(inputTokensArr3[tokenPosVal6])
        Next
        
        ProcesarEntrada(inputIDsArr3)
    EndIf
    
    Return True
End Method
End Type
