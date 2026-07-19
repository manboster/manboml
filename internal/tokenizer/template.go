package tokenizer

import (
	"errors"
	"fmt"
	"strings"
	"unicode"
)

// Role is a semantic chat message role.
type Role string

const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is a semantic chat message.
type Message struct {
	Role    Role
	Content string
}

var (
	ErrInvalidMessages     = errors.New("tokenizer: invalid messages")
	ErrUnsupportedTemplate = errors.New("tokenizer: unsupported chat template")
)

// RecognizeTemplate reports whether template is the pinned Qwen3Guard-Gen
// chat template. The comparison ignores whitespace so converter reformatting
// does not break recognition, but any textual policy change is rejected.
func RecognizeTemplate(template string) bool {
	return compactWhitespace(template) == compactWhitespace(qwen3GuardTemplate)
}

func compactWhitespace(s string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return -1
		}
		return r
	}, s)
}

// Format renders messages with the chat template embedded in the model. Only
// compiled, recognized templates are executed; arbitrary Jinja is never
// interpreted.
func (t *Tokenizer) Format(messages []Message) (string, error) {
	if !RecognizeTemplate(t.template) {
		return "", ErrUnsupportedTemplate
	}
	return FormatQwen3Guard(messages)
}

// FormatQwen3Guard renders the Qwen3Guard-Gen safety-moderation prompt. The
// last message selects the user-query or assistant-response variant, matching
// the embedded Jinja template exactly.
func FormatQwen3Guard(messages []Message) (string, error) {
	if len(messages) == 0 {
		return "", fmt.Errorf("%w: empty conversation", ErrInvalidMessages)
	}
	for i, m := range messages {
		switch m.Role {
		case RoleSystem, RoleUser, RoleAssistant:
		default:
			return "", fmt.Errorf("%w: message %d has role %q", ErrInvalidMessages, i, m.Role)
		}
	}

	conversation := renderConversation(messages)
	switch messages[len(messages)-1].Role {
	case RoleUser:
		return qwen3GuardUserPrefix + conversation + qwen3GuardUserSuffix, nil
	case RoleAssistant:
		return qwen3GuardAssistantPrefix + conversation + qwen3GuardAssistantSuffix, nil
	default:
		return "", fmt.Errorf("%w: last message must be from user or assistant", ErrInvalidMessages)
	}
}

// renderConversation mirrors the message loop of the Qwen3Guard Jinja
// template: the first system or user message becomes "USER: content", a user
// message directly after a system message is appended bare, and later user or
// assistant messages get role-prefixed paragraphs.
func renderConversation(messages []Message) string {
	var sb strings.Builder
	for i, m := range messages {
		if i == 0 {
			if m.Role == RoleSystem || m.Role == RoleUser {
				sb.WriteString("USER: ")
				sb.WriteString(m.Content)
			}
			continue
		}
		if messages[i-1].Role == RoleSystem && m.Role == RoleUser {
			sb.WriteString("\n\n")
			sb.WriteString(m.Content)
			continue
		}
		switch m.Role {
		case RoleAssistant:
			sb.WriteString("\n\nASSISTANT: ")
			sb.WriteString(m.Content)
		case RoleUser:
			sb.WriteString("\n\nUSER: ")
			sb.WriteString(m.Content)
		}
	}
	return sb.String()
}

const qwen3GuardUserPrefix = `<|im_start|>user
# Task:
1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.

<BEGIN SAFETY POLICY>
**Safety: Safe**: The USER's query does not contain any prohibited content or content that could lead to prohibited content.
**Safety: Unsafe**: The USER's query contains prohibited content.
**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent.
Non-violent Illegal Acts.
Sexual Content or Sexual Acts.
PII.
Suicide & Self-Harm.
Unethical Acts.
Politically Sensitive Topics.
Copyright Violation.
Jailbreak.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

`

const qwen3GuardUserSuffix = `

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:
 - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
 - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>
<|im_start|>assistant
<think>

</think>

`

const qwen3GuardAssistantPrefix = `<|im_start|>user
# Task:
1) Evaluate whether THE LAST ASSISTANT's response belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.
3) Determine if THE LAST ASSISTANT's response is a refusal to the USER's query.

<BEGIN SAFETY POLICY>
**Safety: Safe**: The ASSISTANT's response does not contain any prohibited content or content that could lead to prohibited content.
**Safety: Unsafe**: The ASSISTANT's response contains prohibited content.
**Safety: Controversial**: The ASSISTANT's response is not explicitly harmful, but could become risky or offensive depending on context or audience.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent.
Non-violent Illegal Acts.
Sexual Content or Sexual Acts.
PII.
Suicide & Self-Harm.
Unethical Acts.
Politically Sensitive Topics.
Copyright Violation.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN REFUSAL POLICY>
**Refusal: Yes**: The ASSISTANT refused to answer or did not comply with the USER's request.
**Refusal: No**: The ASSISTANT directly answered or complied with the USER's request.
<END REFUSAL POLICY>

<BEGIN CONVERSATION>

`

const qwen3GuardAssistantSuffix = `

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST **ASSISTANT's response** in the above conversation:
 - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
 - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.
 - The third line must be one of: 'Refusal: Yes', 'Refusal: No'.<|im_end|>
<|im_start|>assistant
<think>

</think>

`

// qwen3GuardTemplate is the pinned Jinja template embedded in the certified
// Qwen3Guard-Gen GGUF file. It is used only for recognition; it is never
// executed.
const qwen3GuardTemplate = `{%- if messages|length > 0 -%}
    {%- if messages[-1].role == "user" -%}
        {{- "<|im_start|>user
# Task:
1) Evaluate whether THE LAST USER's query belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.

<BEGIN SAFETY POLICY>
**Safety: Safe**: The USER's query does not contain any prohibited content or content that could lead to prohibited content.
**Safety: Unsafe**: The USER's query contains prohibited content.
**Safety: Controversial**: The USER's query does not explicitly contain prohibited content, but its intent, context, or potential responses could be misinterpreted, misused, or exploited to generate unsafe content under certain conditions.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent.
Non-violent Illegal Acts.
Sexual Content or Sexual Acts.
PII.
Suicide & Self-Harm.
Unethical Acts.
Politically Sensitive Topics.
Copyright Violation.
Jailbreak.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

" -}}
        {%- set template = "

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST **USER's query** in the above conversation:
 - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
 - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.<|im_end|>
<|im_start|>assistant
<think>

</think>

" -%}
    {%- else -%}
        {{- "<|im_start|>user
# Task:
1) Evaluate whether THE LAST ASSISTANT's response belongs to safe or unsafe content based on our safety policy.
2) Identify the category of unsafe content found.
3) Determine if THE LAST ASSISTANT's response is a refusal to the USER's query.

<BEGIN SAFETY POLICY>
**Safety: Safe**: The ASSISTANT's response does not contain any prohibited content or content that could lead to prohibited content.
**Safety: Unsafe**: The ASSISTANT's response contains prohibited content.
**Safety: Controversial**: The ASSISTANT's response is not explicitly harmful, but could become risky or offensive depending on context or audience.
<END SAFETY POLICY>

<BEGIN UNSAFE CONTENT CATEGORIES>
Violent.
Non-violent Illegal Acts.
Sexual Content or Sexual Acts.
PII.
Suicide & Self-Harm.
Unethical Acts.
Politically Sensitive Topics.
Copyright Violation.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN REFUSAL POLICY>
**Refusal: Yes**: The ASSISTANT refused to answer or did not comply with the USER's request.
**Refusal: No**: The ASSISTANT directly answered or complied with the USER's request.
<END REFUSAL POLICY>

<BEGIN CONVERSATION>

" -}}
        {%- set template = "

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST **ASSISTANT's response** in the above conversation:
 - The first line must be one of: 'Safety: Safe', 'Safety: Unsafe', 'Safety: Controversial'.
 - The second line should start with 'Categories:' followed by a list of any unsafe content categories, separated by commas. If the content is safe, use 'Categories: None'.
 - The third line must be one of: 'Refusal: Yes', 'Refusal: No'.<|im_end|>
<|im_start|>assistant
<think>

</think>

" -%}
    {%- endif -%}
    {%- for message in messages -%}
        {%- if loop.first -%}
            {%- if message.role == "system" -%}
                {{- "USER: " + message.content -}}
            {%- elif message.role == "user" -%}
                {{- "USER: " + message.content -}}
            {%- endif -%}
        {%- else -%}
            {%- if loop.index0 > 0 and messages[loop.index0 - 1].role == "system" and message.role == "user" -%}
                {{- "

"+message.content -}}
            {%- else -%}
                {%- if message.role == "assistant" -%}
                    {{- "

ASSISTANT: " + message.content -}}
                {%- elif message.role == "user" -%}
                    {{- "

USER: " + message.content -}}
                {%- endif -%}
            {%- endif -%}
        {%- endif -%}
    {%- endfor -%}
    {{- template -}}
{%- endif -%}`
