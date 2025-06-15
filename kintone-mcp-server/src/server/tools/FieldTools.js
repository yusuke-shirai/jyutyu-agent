// src/server/tools/FieldTools.js
import { UNIT_POSITION_PATTERNS, LOOKUP_FIELD_MIN_WIDTH } from '../../constants.js';
import { ValidationUtils } from '../../utils/ValidationUtils.js';
import { LoggingUtils } from '../../utils/LoggingUtils.js';
import { ResponseBuilder } from '../../utils/ResponseBuilder.js';
import { FieldValidationUtils } from '../../utils/FieldValidationUtils.js';

/**
 * 単位記号に基づいて適切な unitPosition を判定する関数
 * @param {string} unit 単位記号
 * @returns {string} 適切な unitPosition ("BEFORE" または "AFTER")
 */
function determineUnitPosition(unit) {
    // 判定理由を記録する変数
    let reason = "";
    
    // 単位が指定されていない場合
    if (!unit) {
        reason = "単位が指定されていないため";
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { defaultValue: 'AFTER' });
        return "AFTER";
    }
    
    // 単位の長さが4文字以上の場合
    if (unit.length >= 4) {
        reason = `単位の長さが4文字以上 (${unit.length}文字) のため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER' });
        return "AFTER";
    }
    
    // 複合単位の判定（スペースや特殊記号を含む）
    if (/[\s\/\-\+]/.test(unit) || (unit.length > 1 && /[^\w\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]/.test(unit))) {
        reason = `複合単位 "${unit}" と判断されるため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER' });
        return "AFTER";
    }
    
    // 完全一致による判定
    const isBeforeExact = UNIT_POSITION_PATTERNS.BEFORE.includes(unit);
    const isAfterExact = UNIT_POSITION_PATTERNS.AFTER.includes(unit);
    
    // 両方のパターンに一致する場合
    if (isBeforeExact && isAfterExact) {
        reason = `単位 "${unit}" が BEFORE と AFTER の両方のパターンに一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER (優先)' });
        return "AFTER";
    }
    
    // BEFOREパターンに完全一致
    if (isBeforeExact) {
        reason = `単位 "${unit}" が BEFORE パターンに完全一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'BEFORE' });
        return "BEFORE";
    }
    
    // AFTERパターンに完全一致
    if (isAfterExact) {
        reason = `単位 "${unit}" が AFTER パターンに完全一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER' });
        return "AFTER";
    }
    
    // 部分一致による判定（完全一致しない場合のフォールバック）
    const beforeMatches = UNIT_POSITION_PATTERNS.BEFORE.filter(pattern => unit.includes(pattern));
    const afterMatches = UNIT_POSITION_PATTERNS.AFTER.filter(pattern => unit.includes(pattern));
    
    // 両方のパターンに部分一致する場合
    if (beforeMatches.length > 0 && afterMatches.length > 0) {
        reason = `単位 "${unit}" が BEFORE パターン [${beforeMatches.join(', ')}] と AFTER パターン [${afterMatches.join(', ')}] の両方に部分一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER (優先)' });
        return "AFTER";
    }
    
    // BEFOREパターンに部分一致
    if (beforeMatches.length > 0) {
        reason = `単位 "${unit}" が BEFORE パターン [${beforeMatches.join(', ')}] に部分一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'BEFORE' });
        return "BEFORE";
    }
    
    // AFTERパターンに部分一致
    if (afterMatches.length > 0) {
        reason = `単位 "${unit}" が AFTER パターン [${afterMatches.join(', ')}] に部分一致するため`;
        LoggingUtils.logDetailedOperation('単位位置判定', reason, { unitPosition: 'AFTER' });
        return "AFTER";
    }
    
    // どのパターンにも一致しない場合
    reason = `単位 "${unit}" がどのパターンにも一致しないため`;
    LoggingUtils.logDetailedOperation('単位位置判定', reason, { defaultValue: 'AFTER' });
    return "AFTER";
}

/**
 * フィールドの単位位置を自動修正する関数
 * @param {Object} field フィールドオブジェクト
 * @returns {Object} 修正されたフィールドオブジェクト
 */
export function autoCorrectUnitPosition(field) {
    // フィールドオブジェクトのディープコピーを作成
    const correctedField = JSON.parse(JSON.stringify(field));
    
    // NUMBER フィールドの場合
    if (field.type === "NUMBER" && field.unit && !field.unitPosition) {
        // 単位記号に基づいて適切な unitPosition を判定
        correctedField.unitPosition = determineUnitPosition(field.unit);
        LoggingUtils.logDetailedOperation('フィールド修正', `NUMBER フィールドの unitPosition を自動設定`, { fieldCode: field.code, unitPosition: correctedField.unitPosition });
    }
    
    // CALC フィールドの場合
    if (field.type === "CALC" && field.format === "NUMBER" && field.unit && !field.unitPosition) {
        // 単位記号に基づいて適切な unitPosition を判定
        correctedField.unitPosition = determineUnitPosition(field.unit);
        LoggingUtils.logDetailedOperation('フィールド修正', `CALC フィールドの unitPosition を自動設定`, { fieldCode: field.code, unitPosition: correctedField.unitPosition });
    }
    
    // サブテーブルフィールドの場合、内部のフィールドも再帰的に処理
    if (field.type === "SUBTABLE" && field.fields) {
        // 各サブフィールドに対して自動修正を適用
        for (const [fieldKey, fieldDef] of Object.entries(field.fields)) {
            correctedField.fields[fieldKey] = autoCorrectUnitPosition(fieldDef);
        }
        LoggingUtils.logDetailedOperation('フィールド修正', `SUBTABLE フィールド内の単位位置を自動修正`, { fieldCode: field.code });
    }
    
    return correctedField;
}

/**
 * 単位記号と unitPosition の組み合わせが適切かチェックし、警告メッセージを返す関数
 * @param {string} unit 単位記号
 * @param {string} unitPosition 単位位置
 * @returns {string|null} 警告メッセージ（問題がなければ null）
 */
function checkUnitPositionWarning(unit, unitPosition) {
    if (!unit || !unitPosition) return null;
    
    const recommendedPosition = determineUnitPosition(unit);
    
    if (unitPosition !== recommendedPosition) {
        const examples = {
            "BEFORE": "$100, ¥100",
            "AFTER": "100円, 100%, 100kg"
        };
        
        return `単位記号「${unit}」には unitPosition="${recommendedPosition}" が推奨されます。` +
               `現在の設定: "${unitPosition}"。` +
               `例: ${examples[recommendedPosition]}`;
    }
    
    return null;
}

// フィールド関連のツールを処理する関数
export async function handleFieldTools(name, args, repository) {
    // 共通のツール実行ログ
    LoggingUtils.logToolExecution('field', name, args);
    
    switch (name) {
        case 'add_fields': {
            ValidationUtils.validateRequired(args, ['app_id', 'properties']);
            ValidationUtils.validateObject(args.properties, 'properties');
            
            if (Object.keys(args.properties).length === 0) {
                throw new Error('properties には少なくとも1つのフィールド定義を指定する必要があります。');
            }
            
            // レイアウト要素（SPACER, HR, LABEL）のチェック
            const layoutElementTypes = ['SPACER', 'HR', 'LABEL'];
            for (const [key, field] of Object.entries(args.properties)) {
                if (field.type && layoutElementTypes.includes(field.type)) {
                    throw new Error(
                        `レイアウト要素 "${field.type}" は add_fields ツールではサポートされていません。\n\n` +
                        `スペース、罫線、ラベルなどのレイアウト要素は、フォームのレイアウト設定で追加する必要があります。\n\n` +
                        `【代替方法】\n` +
                        `1. update_form_layout ツールを使用してフォームレイアウトを更新する\n` +
                        `2. add_layout_element ツールを使用して特定の位置にレイアウト要素を追加する\n\n` +
                        `【使用例】\n` +
                        `// スペース要素を作成\n` +
                        `const spacerElement = {\n` +
                        `  type: "SPACER",\n` +
                        `  elementId: "spacer1",\n` +
                        `  size: { width: 100, height: 30 }\n` +
                        `};\n\n` +
                        `// レイアウトに要素を追加\n` +
                        `add_layout_element({\n` +
                        `  app_id: ${args.app_id},\n` +
                        `  element: spacerElement\n` +
                        `});\n\n` +
                        `または、create_spacer_element、create_hr_element、create_label_element ツールを使用して\n` +
                        `簡単にレイアウト要素を作成することもできます。`
                    );
                }
            }
            
            // フィールドのコード設定を確認・修正
            const processedProperties = {};
            for (const [key, field] of Object.entries(args.properties)) {
                // フィールドコードが指定されていない場合は自動生成
                if (!field.code) {
                    // ラベルがある場合はラベルからコードを生成、なければキーを使用
                    const baseCode = field.label ? field.label : key;
                    // 英数字以外を削除し、小文字に変換
                    const code = baseCode
                        .replace(/[^a-zA-Z0-9ぁ-んァ-ヶー一-龠々＿_･・＄￥]/g, '')
                        .toLowerCase();
                    field.code = code || `field_${Date.now()}`;
                    LoggingUtils.logDetailedOperation('フィールド処理', 'フィールドコードを自動生成', { code: field.code });
                }
                
                // フィールドタイプが指定されていない場合はエラー
                if (!field.type) {
                    throw new Error(`フィールド "${field.code}" にはタイプ(type)の指定が必須です。`);
                }
                
                // 計算フィールドの場合、formulaからexpressionへの自動変換
                if (field.type === "CALC" && field.formula !== undefined && field.expression === undefined) {
                    field.expression = field.formula;
                    delete field.formula;
                    LoggingUtils.logWarning('add_fields', `計算フィールド "${field.code}" の計算式は formula ではなく expression に指定してください。今回は自動的に変換しました。`);
                }
                
                // 数値フィールドの場合、displayScaleが空文字列なら削除
                if (field.type === "NUMBER" && field.displayScale === "") {
                    delete field.displayScale;
                    LoggingUtils.logWarning('add_fields', `数値フィールド "${field.code}" の displayScale に空文字列が指定されたため、指定を削除しました。`);
                }
                
                processedProperties[field.code] = field;
            }
            
            const response = await repository.addFields(
                args.app_id,
                processedProperties
            );
            
            return ResponseBuilder.withRevisionAndWarnings(response.revision, response.warnings);
        }
        
        case 'create_choice_field': {
            ValidationUtils.validateRequired(args, ['field_type', 'label', 'choices']);
            ValidationUtils.validateArray(args.choices, 'choices');
            
            // 有効なフィールドタイプかチェック
            const validFieldTypes = ["RADIO_BUTTON", "CHECK_BOX", "DROP_DOWN", "MULTI_SELECT"];
            if (!validFieldTypes.includes(args.field_type)) {
                throw new Error(`field_type は ${validFieldTypes.join(', ')} のいずれかである必要があります。`);
            }
            
            // フィールドコードの自動生成
            let code = args.code;
            if (!code) {
                // ラベルからコードを生成
                code = args.label
                    .replace(/[^a-zA-Z0-9ぁ-んァ-ヶー一-龠々＿_･・＄￥]/g, '_')
                    .toLowerCase();
                
                // 先頭が数字の場合、先頭に 'f_' を追加
                if (/^[0-9０-９]/.test(code)) {
                    code = 'f_' + code;
                }
                
                LoggingUtils.logDetailedOperation('create_choice_field', 'フィールドコードを自動生成', { code });
            }
            
            const { field_type, label, choices, required = false, align = "HORIZONTAL" } = args;
            
            // options オブジェクトの生成
            const options = {};
            choices.forEach((choice, index) => {
                options[choice] = {
                    label: choice,
                    index: String(index)
                };
            });
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: field_type,
                code: code,
                label: label,
                noLabel: false,
                required: required,
                options: options
            };
            
            // フィールドタイプ固有の設定を追加
            if (field_type === "RADIO_BUTTON") {
                fieldConfig.defaultValue = choices.length > 0 ? choices[0] : "";
                fieldConfig.align = align;
            } else if (field_type === "CHECK_BOX") {
                fieldConfig.defaultValue = [];
                fieldConfig.align = align;
            } else if (field_type === "MULTI_SELECT") {
                fieldConfig.defaultValue = [];
            } else if (field_type === "DROP_DOWN") {
                fieldConfig.defaultValue = "";
            }
            
            return fieldConfig;
        }
        
        case 'create_reference_table_field': {
            ValidationUtils.validateRequired(args, ['label', 'conditionField', 'relatedConditionField']);
            
            if (!args.relatedAppId && !args.relatedAppCode) {
                throw new Error('relatedAppId または relatedAppCode のいずれかは必須パラメータです。');
            }
            
            // フィールドコードの自動生成
            let code = args.code;
            if (!code) {
                // ラベルからコードを生成
                code = args.label
                    .replace(/[^a-zA-Z0-9ぁ-んァ-ヶー一-龠々＿_･・＄￥]/g, '_')
                    .toLowerCase();
                
                // 先頭が数字の場合、先頭に 'f_' を追加
                if (/^[0-9０-９]/.test(code)) {
                    code = 'f_' + code;
                }
                
                LoggingUtils.logDetailedOperation('create_reference_table_field', 'フィールドコードを自動生成', { code });
            }
            
            const { 
                label, 
                relatedAppId, 
                relatedAppCode, 
                conditionField, 
                relatedConditionField, 
                filterCond, 
                displayFields, 
                sort, 
                size, 
                noLabel = true 
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "REFERENCE_TABLE",
                code: code,
                label: label,
                noLabel: noLabel,
                referenceTable: {
                    relatedApp: {},
                    condition: {
                        field: conditionField,
                        relatedField: relatedConditionField
                    }
                }
            };
            
            // relatedApp の設定（app と code の優先順位に注意）
            if (relatedAppCode) {
                fieldConfig.referenceTable.relatedApp.code = relatedAppCode;
            }
            if (relatedAppId && !relatedAppCode) {
                fieldConfig.referenceTable.relatedApp.app = relatedAppId;
            }
            
            // オプション項目の追加
            if (filterCond) fieldConfig.referenceTable.filterCond = filterCond;
            if (displayFields && Array.isArray(displayFields)) fieldConfig.referenceTable.displayFields = displayFields;
            if (sort) fieldConfig.referenceTable.sort = sort;
            if (size) fieldConfig.referenceTable.size = String(size); // 文字列型に変換
            
            return fieldConfig;
        }
        
        case 'create_lookup_field': {
            ValidationUtils.validateRequired(args, ['label', 'relatedKeyField']);
            
            if (!args.relatedAppId && !args.relatedAppCode) {
                throw new Error('relatedAppId または relatedAppCode のいずれかは必須パラメータです。');
            }
            
            // フィールドコードの自動生成
            let code = args.code;
            if (!code) {
                // ラベルからコードを生成
                code = args.label
                    .replace(/[^a-zA-Z0-9ぁ-んァ-ヶー一-龠々＿_･・＄￥]/g, '_')
                    .toLowerCase();
                
                // 先頭が数字の場合、先頭に 'f_' を追加
                if (/^[0-9０-９]/.test(code)) {
                    code = 'f_' + code;
                }
                
                LoggingUtils.logDetailedOperation('create_lookup_field', 'フィールドコードを自動生成', { code });
            }
            
            const { 
                label, 
                relatedAppId, 
                relatedAppCode, 
                relatedKeyField, 
                fieldMappings, 
                lookupPickerFields, 
                filterCond, 
                sort, 
                required = false,
                fieldType = "SINGLE_LINE_TEXT" // デフォルトのフィールドタイプ
            } = args;
            
            // バリデーション
            ValidationUtils.validateArray(fieldMappings, 'fieldMappings', { minLength: 1 });
            
            // フィールドマッピングの各要素をチェック
            fieldMappings.forEach((mapping, index) => {
                if (!mapping.field) {
                    throw new Error(`fieldMappings[${index}].fieldは必須です`);
                }
                if (!mapping.relatedField) {
                    throw new Error(`fieldMappings[${index}].relatedFieldは必須です`);
                }
                
                // ルックアップのキー自体がマッピングに含まれていないかチェック
                if (mapping.relatedField === relatedKeyField) {
                    throw new Error(`ルックアップのキーフィールド "${relatedKeyField}" はフィールドマッピングに含めないでください`);
                }
            });
            
            // lookupPickerFieldsのチェック
            if (!lookupPickerFields || !Array.isArray(lookupPickerFields) || lookupPickerFields.length === 0) {
                LoggingUtils.logWarning('create_lookup_field', 'lookupPickerFieldsが指定されていません。ルックアップピッカーに表示するフィールドを指定することを推奨します。');
            }
            
            // sortのチェック
            if (!sort) {
                LoggingUtils.logWarning('create_lookup_field', 'sortが指定されていません。ルックアップの検索結果のソート順を指定することを推奨します。');
            }
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: fieldType || "SINGLE_LINE_TEXT", // fieldTypeが指定されていない場合はデフォルト値を使用
                code: code,
                label: label,
                required: required,
                lookup: {
                    relatedApp: {},
                    relatedKeyField: relatedKeyField,
                    fieldMappings: fieldMappings
                }
            };
            
            // 幅の設定と補正
            // 幅が指定されていない場合、または指定された幅が最小幅より小さい場合は最小幅を設定
            if (!args.size) {
                fieldConfig.size = { width: LOOKUP_FIELD_MIN_WIDTH };
                LoggingUtils.logDetailedOperation('create_lookup_field', 'ルックアップフィールドの幅を最小幅に設定', { code, minWidth: LOOKUP_FIELD_MIN_WIDTH });
            } else if (args.size) {
                fieldConfig.size = { ...args.size };
                if (!fieldConfig.size.width || parseInt(fieldConfig.size.width, 10) < parseInt(LOOKUP_FIELD_MIN_WIDTH, 10)) {
                    fieldConfig.size.width = LOOKUP_FIELD_MIN_WIDTH;
                    LoggingUtils.logDetailedOperation('create_lookup_field', 'ルックアップフィールドの幅を最小幅に補正', { code, minWidth: LOOKUP_FIELD_MIN_WIDTH });
                }
            }
            
            // relatedApp の設定（code が優先）
            if (relatedAppCode) {
                fieldConfig.lookup.relatedApp.code = relatedAppCode;
            }
            if (relatedAppId && !relatedAppCode) {
                fieldConfig.lookup.relatedApp.app = relatedAppId;
            }
            
            // オプション項目の追加
            if (lookupPickerFields && Array.isArray(lookupPickerFields)) {
                fieldConfig.lookup.lookupPickerFields = lookupPickerFields;
            } else {
                // デフォルトのlookupPickerFieldsを設定
                // 少なくともキーフィールドは含める
                fieldConfig.lookup.lookupPickerFields = [relatedKeyField];
                LoggingUtils.logDetailedOperation('create_lookup_field', 'lookupPickerFieldsのデフォルト値を設定', { defaultValue: [relatedKeyField] });
            }
            
            if (filterCond) fieldConfig.lookup.filterCond = filterCond;
            
            if (sort) {
                fieldConfig.lookup.sort = sort;
            } else {
                // デフォルトのsortを設定
                fieldConfig.lookup.sort = `${relatedKeyField} asc`;
                LoggingUtils.logDetailedOperation('create_lookup_field', 'sortのデフォルト値を設定', { defaultValue: `${relatedKeyField} asc` });
            }
            
            return fieldConfig;
        }
        
        case 'create_text_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'field_type']);
            ValidationUtils.validateString(args.field_type, 'field_type', {
                allowedValues: ['SINGLE_LINE_TEXT', 'MULTI_LINE_TEXT']
            });
            
            const { 
                field_type, 
                code, 
                label, 
                required = false, 
                noLabel = false,
                unique = false,
                maxLength,
                minLength,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: field_type,
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            if (maxLength !== undefined) fieldConfig.maxLength = String(maxLength);
            if (minLength !== undefined) fieldConfig.minLength = String(minLength);
            
            return fieldConfig;
        }
        
        case 'create_number_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false,
                unique = false,
                maxValue,
                minValue,
                defaultValue = "",
                digit = false,
                unit,
                unitPosition,
                displayScale
            } = args;
            
            // 単位記号に基づいて適切な unitPosition を判定
            let effectiveUnitPosition;
            
            if (unitPosition !== undefined) {
                // ユーザーが明示的に指定した場合はその値を使用
                effectiveUnitPosition = unitPosition;
                
                // 単位記号と unitPosition の組み合わせが不自然な場合は警告
                if (unit) {
                    const warning = checkUnitPositionWarning(unit, unitPosition);
                    if (warning) {
                        LoggingUtils.logWarning('create_number_field', warning);
                    }
                }
            } else if (unit) {
                // 単位記号が指定されている場合は自動判定
                effectiveUnitPosition = determineUnitPosition(unit);
                LoggingUtils.logDetailedOperation('create_number_field', 'unitPositionを自動設定', { unit, unitPosition: effectiveUnitPosition });
            } else {
                // どちらも指定されていない場合はデフォルト値を AFTER に変更
                effectiveUnitPosition = "AFTER";
            }
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "NUMBER",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            if (digit) fieldConfig.digit = digit;
            if (maxValue !== undefined) fieldConfig.maxValue = String(maxValue);
            if (minValue !== undefined) fieldConfig.minValue = String(minValue);
            if (unit !== undefined) fieldConfig.unit = unit;
            fieldConfig.unitPosition = effectiveUnitPosition;
            
            // displayScaleが空文字列なら削除、それ以外は設定
            if (displayScale === "") {
                LoggingUtils.logWarning('create_number_field', `数値フィールド "${code}" の displayScale に空文字列が指定されたため、指定を削除しました。`);
                // displayScaleを設定しない
            } else if (displayScale !== undefined) {
                // displayScaleの値の範囲チェック
                const scale = parseInt(displayScale, 10);
                if (isNaN(scale) || scale < 0 || scale > 10) {
                    throw new Error('displayScaleは0から10までの整数で指定してください。');
                }
                fieldConfig.displayScale = String(displayScale);
            }
            
            return fieldConfig;
        }
        
        case 'create_date_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false,
                unique = false,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "DATE",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            
            return fieldConfig;
        }
        
        case 'create_time_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false,
                unique = false,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "TIME",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            
            return fieldConfig;
        }
        
        case 'create_datetime_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false,
                unique = false,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "DATETIME",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            
            return fieldConfig;
        }
        
        case 'create_rich_text_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "RICH_TEXT",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            
            return fieldConfig;
        }
        
        case 'create_attachment_field': {
            ValidationUtils.validateRequired(args, ['code', 'label']);
            
            const { 
                code, 
                label, 
                required = false, 
                noLabel = false
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "FILE",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            return fieldConfig;
        }
        
        case 'create_user_select_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'field_type']);
            ValidationUtils.validateString(args.field_type, 'field_type', {
                allowedValues: ['USER_SELECT', 'GROUP_SELECT', 'ORGANIZATION_SELECT']
            });
            
            const { 
                field_type,
                code, 
                label, 
                required = false, 
                noLabel = false,
                defaultValue = [],
                entities = []
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: field_type,
                code: code,
                label: label,
                noLabel: noLabel,
                required: required
            };
            
            // オプション項目の追加
            if (defaultValue.length > 0) fieldConfig.defaultValue = defaultValue;
            if (entities.length > 0) fieldConfig.entities = entities;
            
            return fieldConfig;
        }
        
        case 'create_subtable_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'fields']);
            ValidationUtils.validateArray(args.fields, 'fields', { minLength: 1 });
            
            const { 
                code, 
                label, 
                fields
            } = args;
            
            // テーブル内のフィールド定義を構築
            const subtableFields = {};
            
            // 各フィールド定義を処理
            for (const fieldDef of fields) {
                if (!fieldDef.code) {
                    throw new Error('テーブル内の各フィールドには code の指定が必須です。');
                }
                if (!fieldDef.type) {
                    throw new Error(`テーブル内のフィールド "${fieldDef.code}" には type の指定が必須です。`);
                }
                if (!fieldDef.label) {
                    throw new Error(`テーブル内のフィールド "${fieldDef.code}" には label の指定が必須です。`);
                }
                
                // テーブル内では使用できないフィールドタイプをチェック
                const invalidSubtableFieldTypes = [
                    'SUBTABLE',
                    'REFERENCE_TABLE',
                    'STATUS',
                    'RELATED_RECORDS',
                    'RECORD_NUMBER',
                    'CREATOR',
                    'MODIFIER',
                    'CREATED_TIME',
                    'UPDATED_TIME'
                ];
                
                if (invalidSubtableFieldTypes.includes(fieldDef.type)) {
                    throw new Error(`テーブル内では "${fieldDef.type}" タイプのフィールドは使用できません。`);
                }
                
                // フィールド定義をテーブルのfieldsに追加
                subtableFields[fieldDef.code] = fieldDef;
            }
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "SUBTABLE",
                code: code,
                label: label,
                noLabel: false,
                fields: subtableFields
            };
            
            return fieldConfig;
        }
        
        case 'create_calc_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'expression']);
            
            const { 
                code, 
                label, 
                expression,
                noLabel = false,
                // 表示形式関連パラメータ
                format,
                digit = false,
                displayScale = "0",
                unit = "",
                unitPosition
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "CALC",
                code: code,
                label: label,
                noLabel: noLabel,
                expression: expression
            };
            
            // 表示形式の設定
            // digit=trueの場合はNUMBER_DIGITを使用、それ以外はformatパラメータまたはデフォルト値を使用
            if (digit === true) {
                fieldConfig.format = "NUMBER_DIGIT";
                LoggingUtils.logDetailedOperation('create_calc_field', 'formatをNUMBER_DIGITに設定', { reason: '桁区切り表示が有効' });
            } else if (format) {
                fieldConfig.format = format;
            } else {
                // デフォルトでNUMBER_DIGITを使用（桁区切り表示をデフォルトにする）
                fieldConfig.format = "NUMBER_DIGIT";
                LoggingUtils.logDetailedOperation('create_calc_field', 'formatのデフォルト値を設定', { defaultValue: 'NUMBER_DIGIT' });
            }
            
            // 数値形式の場合の追加設定
            if (fieldConfig.format === "NUMBER" || fieldConfig.format === "NUMBER_DIGIT") {
                // digitはformatで表現するため、fieldConfigには含めない
                
                if (displayScale !== undefined) fieldConfig.displayScale = String(displayScale);
                if (unit !== undefined) fieldConfig.unit = unit;
                
                // 単位記号に基づいて適切な unitPosition を判定
                let effectiveUnitPosition;
                
                if (unitPosition !== undefined) {
                    // ユーザーが明示的に指定した場合はその値を使用
                    effectiveUnitPosition = unitPosition;
                    
                    // 単位記号と unitPosition の組み合わせが不自然な場合は警告
                    if (unit) {
                        const warning = checkUnitPositionWarning(unit, unitPosition);
                        if (warning) {
                            LoggingUtils.logWarning('create_calc_field', warning);
                        }
                    }
                } else if (unit) {
                    // 単位記号が指定されている場合は自動判定
                    effectiveUnitPosition = determineUnitPosition(unit);
                    LoggingUtils.logDetailedOperation('create_calc_field', 'unitPositionを自動設定', { unit, unitPosition: effectiveUnitPosition });
                } else {
                    // どちらも指定されていない場合はデフォルト値を AFTER に変更
                    effectiveUnitPosition = "AFTER";
                }
                
                fieldConfig.unitPosition = effectiveUnitPosition;
            }
            
            return fieldConfig;
        }
        
        case 'create_status_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'states']);
            ValidationUtils.validateArray(args.states, 'states', { minLength: 1 });
            
            const { 
                code, 
                label, 
                states,
                defaultState = states[0].name,
                noLabel = false
            } = args;
            
            // 状態定義を構築
            const statesObj = {};
            
            // 各状態定義を処理
            for (const stateDef of states) {
                if (!stateDef.name) {
                    throw new Error('各状態には name の指定が必須です。');
                }
                
                // 状態定義をstatesに追加
                statesObj[stateDef.name] = {
                    name: stateDef.name,
                    index: stateDef.index || String(Object.keys(statesObj).length),
                    ...(stateDef.transitions && { transitions: stateDef.transitions })
                };
            }
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "STATUS",
                code: code,
                label: label,
                noLabel: noLabel,
                states: statesObj,
                defaultState: defaultState
            };
            
            return fieldConfig;
        }
        
        case 'create_related_records_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'relatedApp', 'condition']);
            ValidationUtils.validateObject(args.condition, 'condition');
            ValidationUtils.validateRequired(args.condition, ['field', 'relatedField']);
            
            const { 
                code, 
                label, 
                relatedApp,
                condition,
                displayFields,
                filterCond,
                sort,
                noLabel = false
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "RELATED_RECORDS",
                code: code,
                label: label,
                noLabel: noLabel,
                relatedApp: relatedApp,
                condition: condition
            };
            
            // オプション項目の追加
            if (displayFields) fieldConfig.displayFields = displayFields;
            if (filterCond) fieldConfig.filterCond = filterCond;
            if (sort) fieldConfig.sort = sort;
            
            return fieldConfig;
        }
        
        case 'create_link_field': {
            ValidationUtils.validateRequired(args, ['code', 'label', 'protocol']);
            ValidationUtils.validateString(args.protocol, 'protocol', {
                allowedValues: ['WEB', 'MAIL', 'CALL']
            });
            
            const { 
                code, 
                label, 
                protocol,
                required = false, 
                noLabel = false,
                unique = false,
                maxLength,
                minLength,
                defaultValue = ""
            } = args;
            
            // フィールド設定の基本部分
            const fieldConfig = {
                type: "LINK",
                code: code,
                label: label,
                noLabel: noLabel,
                required: required,
                protocol: protocol
            };
            
            // オプション項目の追加
            if (unique) fieldConfig.unique = unique;
            if (defaultValue !== "") fieldConfig.defaultValue = defaultValue;
            if (maxLength !== undefined) fieldConfig.maxLength = String(maxLength);
            if (minLength !== undefined) fieldConfig.minLength = String(minLength);
            
            return fieldConfig;
        }
        
        case 'update_field': {
            ValidationUtils.validateRequired(args, ['app_id', 'field_code', 'field']);
            ValidationUtils.validateObject(args.field, 'field');
            
            // フィールドのタイプチェック
            if (!args.field.type) {
                throw new Error(`フィールド "${args.field_code}" にはタイプ(type)の指定が必須です。`);
            }
            
            // システムフィールドタイプのチェック
            const systemFieldTypes = ['RECORD_NUMBER', 'CREATOR', 'MODIFIER', 'CREATED_TIME', 'UPDATED_TIME'];
            if (systemFieldTypes.includes(args.field.type)) {
                throw new Error(
                    `フィールドタイプ "${args.field.type}" は更新できません。これはkintoneによって自動的に作成されるシステムフィールドです。\n` +
                    `代替方法として、以下のようなフィールドを追加できます：\n` +
                    `- CREATOR（作成者）の代わりに「申請者」などの名前でUSER_SELECTフィールド\n` +
                    `- MODIFIER（更新者）の代わりに「承認者」などの名前でUSER_SELECTフィールド\n` +
                    `- CREATED_TIME（作成日時）の代わりに「申請日時」などの名前でDATETIMEフィールド\n` +
                    `- UPDATED_TIME（更新日時）の代わりに「承認日時」などの名前でDATETIMEフィールド\n` +
                    `- RECORD_NUMBER（レコード番号）の代わりに「管理番号」などの名前でSINGLE_LINE_TEXTフィールド`
                );
            }
            
            // プロパティオブジェクトの作成
            const properties = {
                [args.field_code]: args.field
            };
            
            // フィールドの更新
            const response = await repository.updateFormFields(
                args.app_id,
                properties,
                args.revision || -1
            );
            
            return ResponseBuilder.withRevision(response.revision);
        }
        
        default:
            throw new Error(`Unknown field tool: ${name}`);
    }
}
