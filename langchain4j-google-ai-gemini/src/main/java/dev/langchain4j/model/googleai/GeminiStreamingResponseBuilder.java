package dev.langchain4j.model.googleai;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

import static dev.langchain4j.model.googleai.FinishReasonMapper.fromGFinishReasonToFinishReason;
import static dev.langchain4j.model.googleai.PartsAndContentsMapper.fromGPartsToAiMessage;

public class GeminiStreamingResponseBuilder {
    private final boolean includeCodeExecutionOutput;

    private final StringBuffer contentBuilder = new StringBuffer();
    private final List<ToolExecutionRequest> functionCalls = new ArrayList<>();

    private TokenUsage tokenUsage;
    private FinishReason finishReason;

    GeminiStreamingResponseBuilder(boolean includeCodeExecutionOutput){
        this.includeCodeExecutionOutput = includeCodeExecutionOutput;
    }

    String append(GeminiGenerateContentResponse partialResponse) {
        if (partialResponse == null) {
            return null;
        }

        GeminiCandidate firstCandidate = partialResponse.getCandidates().get(0); //TODO handle n
        GeminiUsageMetadata tokenCounts = partialResponse.getUsageMetadata();
        GeminiContent content = firstCandidate.getContent();

        this.tokenUsage = new TokenUsage(
                tokenCounts.getPromptTokenCount(),
                tokenCounts.getCandidatesTokenCount(),
                tokenCounts.getTotalTokenCount()
        );

        if(firstCandidate.getFinishReason() != null){
         this.finishReason = fromGFinishReasonToFinishReason(firstCandidate.getFinishReason());
        }

        if(content == null){
            return null;
        }

        AiMessage message = fromGPartsToAiMessage(content.getParts(), this.includeCodeExecutionOutput);

        if(message.text() != null){
            contentBuilder.append(message.text());
        }

        if(message.hasToolExecutionRequests()){
            functionCalls.addAll(message.toolExecutionRequests());
        }

        return message.text();
    }

    Response<AiMessage> build() {
        AiMessage aiMessage = new AiMessage("No valid return.");
        String text = contentBuilder.toString();
        boolean hasText = !text.isEmpty() && !text.isBlank();

        if(hasText && !functionCalls.isEmpty()){
            aiMessage = new AiMessage(text, functionCalls);
        } else if (hasText) {
            aiMessage = new AiMessage(text);
        } else if (!functionCalls.isEmpty()) {
            aiMessage = new AiMessage(functionCalls);
        }

        return Response.from(aiMessage, tokenUsage, finishReason);
    }
}
