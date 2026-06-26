const express = require('express');
const fs = require('fs');
// API Key などの環境変数は .env.local から読み込む
require('dotenv').config({ path: '.env.local' });

const app = express();
const PORT = process.env.PORT || 8080;

app.use(express.json());
app.use(express.static('public'));

// ===== 設定 =====
// 利用するLLMプロバイダを選択します（'openai' または 'gemini'）
const PROVIDER = process.env.LLM_PROVIDER || 'openai';

// プロバイダごとに利用するモデル
const MODELS = {
    openai: process.env.OPENAI_MODEL || 'gpt-4o-mini',
    gemini: process.env.GEMINI_MODEL || 'gemini-1.5-flash',
};
const MODEL = MODELS[PROVIDER];

let promptTemplate;
try {
    promptTemplate = fs.readFileSync('prompt.md', 'utf8');
} catch (error) {
    console.error('Error reading prompt.md:', error);
    process.exit(1);
}

const OPENAI_API_ENDPOINT = 'https://api.openai.com/v1/chat/completions';
const GEMINI_API_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta/models/';
const APP_VERSION = 'movie-game-2026-06-19-debug';

// public/ 内の .html 一覧を返す（index.html がこの一覧を使ってリンクを表示する）
app.get('/api/pages', (req, res) => {
    const files = fs.readdirSync('public')
        .filter(name => name.endsWith('.html') && name !== 'index.html');
    res.json(files);
});

app.get('/api/health', (req, res) => {
    res.json({
        ok: true,
        version: APP_VERSION,
        provider: PROVIDER,
        model: MODEL,
        hasOpenAIKey: Boolean(process.env.OPENAI_API_KEY),
        hasGeminiKey: Boolean(process.env.GEMINI_API_KEY),
    });
});

// 問題数の上限（過剰なリクエストでトークンを浪費しないようにする）
const MAX_COUNT = 20;
const JSON_RETRY_INSTRUCTION = `

重要: 直前の出力はJSON.parse()に失敗しました。
もう一度、必ず完全なJSONオブジェクトだけを返してください。
各文字列は短くし、reason、answerCommentary、whyRecommended は30文字以内にしてください。
問題文は映画知識がなくても推測しやすい、分かりやすい内容にしてください。
問題文には、その映画ならではの特徴を3つ以上入れてください。ただしタイトルや結末は書かないでください。
配列の区切りカンマ、閉じ括弧、閉じ波括弧を必ず正しく付けてください。
Markdown、コードフェンス、説明文は一切出力しないでください。`;

app.post('/api/', async (req, res) => {
    try {
        // title と、変数置換に使うその他のキーを受け取る
        // （prompt.md がプロンプトを定義するので、リクエストでの上書きは許可しない）
        const { title = 'Generated Content', ...variables } = req.body;

        // count が指定されている場合は 1〜MAX_COUNT の範囲に収める
        if (variables.count !== undefined) {
            const count = Number(variables.count);
            if (!Number.isInteger(count) || count < 1 || count > MAX_COUNT) {
                return res.status(400).json({
                    error: `count must be an integer between 1 and ${MAX_COUNT}`,
                });
            }
        }

        // prompt.md のテンプレート変数 ${key} をリクエストの値で置換する
        const finalPrompt = fillTemplate(promptTemplate, variables);

        let jsonText = await callProvider(finalPrompt);
        let parsedData;
        try {
            parsedData = parseGeneratedJson(jsonText);
        } catch (parseError) {
            console.warn('Retrying after JSON parse failure:', parseError.message);
            jsonText = await callProvider(finalPrompt + JSON_RETRY_INSTRUCTION);
            parsedData = parseGeneratedJson(jsonText);
        }

        res.json({
            title: title,
            data: parsedData,
            jsonText,
            version: APP_VERSION,
        });

    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({
            error: error.message || 'Failed to generate content. Please try again.',
            provider: PROVIDER,
            model: MODEL,
            version: APP_VERSION,
        });
    }
});

async function callProvider(prompt) {
    if (PROVIDER === 'openai') {
        return callOpenAI(prompt);
    }

    if (PROVIDER === 'gemini') {
        return callGemini(prompt);
    }

    throw new Error('Invalid provider configuration');
}

// prompt.md 内の ${key} を variables の値で安全に置換する
function fillTemplate(template, variables) {
    return template.replace(/\$\{(\w+)\}/g, (match, key) => {
        return Object.prototype.hasOwnProperty.call(variables, key)
            ? String(variables[key])
            : match; // 対応する値がなければそのまま残す
    });
}

async function callOpenAI(prompt) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
        throw new Error('OPENAI_API_KEY environment variable is not set');
    }

    const response = await fetch(OPENAI_API_ENDPOINT, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${apiKey}`
        },
        body: JSON.stringify({
            model: MODEL,
            messages: [
                { role: 'system', content: prompt }
            ],
            max_tokens: 6000,
            response_format: { type: "json_object" }
        })
    });

    if (!response.ok) {
        throw new Error(await readApiError(response, 'OpenAI API error'));
    }

    const data = await response.json();
    const responseText = data.choices[0].message.content;
    return normalizeJsonText(responseText);
}

async function callGemini(prompt) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
        throw new Error('GEMINI_API_KEY environment variable is not set');
    }

    const response = await fetch(`${GEMINI_API_BASE_URL}${MODEL}:generateContent?key=${apiKey}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            contents: [{
                parts: [{ text: prompt }]
            }],
            generationConfig: {
                maxOutputTokens: 6000,
                response_mime_type: "application/json"
            }
        })
    });

    if (!response.ok) {
        throw new Error(await readApiError(response, 'Gemini API error'));
    }

    const data = await response.json();
    const responseText = data.candidates[0].content.parts[0].text;
    return normalizeJsonText(responseText);
}

// LLM が返した JSON 文字列をパースし、クライアントへ返す前に形式不正を検出する
function parseGeneratedJson(responseText) {
    try {
        return JSON.parse(responseText);
    } catch (parseError) {
        throw new Error('Failed to parse LLM response: ' + parseError.message);
    }
}

function normalizeJsonText(responseText) {
    const text = String(responseText || '').trim();
    if (!text) {
        throw new Error('LLM response is empty');
    }

    const fencedJson = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/i);
    if (fencedJson) {
        return fencedJson[1].trim();
    }

    const firstBrace = text.indexOf('{');
    const lastBrace = text.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
        return text.slice(firstBrace, lastBrace + 1).trim();
    }

    return text;
}

async function readApiError(response, fallbackMessage) {
    const text = await response.text();
    if (!text) {
        return `${fallbackMessage} (${response.status})`;
    }

    try {
        const json = JSON.parse(text);
        return json.error?.message || json.error || text;
    } catch (parseError) {
        return `${fallbackMessage} (${response.status}): ${text.slice(0, 300)}`;
    }
}

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
    console.log(`Config: ${PROVIDER} - ${MODEL}`);
});
